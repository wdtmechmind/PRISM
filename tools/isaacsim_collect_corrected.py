#!/usr/bin/env python3
"""Isaac Sim data collection in corrected-frame world coordinates.

This script reuses the corrected-frame replay environment, adds a fixed overhead
camera, and records:
1) rendered video from that camera
2) UR5 end-effector trajectory CSV

Usage:
    /isaac-sim/python.sh tools/isaacsim_collect_corrected.py \
        --trial-dir data/raw/task_xxx/trial_000001 \
        --calib-json configs/devices/charuco_4cam_result.json
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# Make sure "src/" is importable when running from repository root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Kept for compatibility with corrected-frame utilities used by sibling tools.

LED_COLORS = {
    "red": (1.0, 0.05, 0.05),
    "yellow": (1.0, 0.9, 0.05),
    "blue": (0.1, 0.3, 1.0),
    "green": (0.1, 0.9, 0.2),
}


def _rotation_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
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
    return q / max(np.linalg.norm(q), 1e-12)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
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
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
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

        ts, pos, rot = [], [], []
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
            rot.append(r)

    if not ts:
        raise RuntimeError("no valid pose rows in %s" % path)

    order = np.argsort(ts)
    t_arr = np.asarray(ts, dtype=np.float64)[order]
    pos_arr = np.asarray(pos, dtype=np.float64)[order]
    rot_arr = np.asarray(rot, dtype=np.float64)[order]
    return t_arr, pos_arr, rot_arr, use_sm


def load_leds(path: str):
    if not path or not os.path.isfile(path):
        return {}
    rows_by_color = {}
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
            parts = ["%s=%s" % (c, row[c]) for c in label_cols if str(row.get(c, "")).strip()]
            if parts:
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
        raise SystemExit("rigid pose CSV not found: %r (use --rigid or --trial-dir)" % rigid)
    return rigid, led, gestures


def sample_pose(t_arr, pos_arr, quat_arr, t):
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


def transform_to_corrected_frame(pos_arr, rot_arr, leds, calib_json):
    cameras = load_calibration(calib_json)
    centers = get_camera_centers_world(cameras)
    corr = build_corrected_transform(cameras, centers)

    pos_corr = apply_corrected_transform(pos_arr, corr)
    rot_world_to_corr = np.asarray(corr["R"], dtype=np.float64).reshape(3, 3)
    rot_corr = np.asarray([rot_world_to_corr @ r for r in rot_arr], dtype=np.float64)

    leds_corr = {}
    for name, pts in leds.items():
        leds_corr[name] = apply_corrected_transform(pts, corr)

    return pos_corr, rot_corr, leds_corr


def make_point_marker(stage, path, position_xyz, color_rgb, radius=0.012):
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])
    xform = UsdGeom.Xformable(sphere)
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position_xyz)))
    return sphere


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial-dir", help="trial dir containing trajectory/ and hand/")
    ap.add_argument("--rigid", help="path to rigid_pose_6d.csv")
    ap.add_argument("--led", help="path to trajectory_led.csv (long format)")
    ap.add_argument("--gestures", help="path to sdk_commands.csv / gesture timeline CSV")
    ap.add_argument("--calib-json", help="kept for CLI compatibility; not required by random touch task")
    ap.add_argument("--speed", type=float, default=1.0, help="motion speed multiplier")
    ap.add_argument("--fps", type=float, default=60.0, help="render frame-rate cap (0 = uncapped)")
    ap.add_argument("--loop", action="store_true", help="loop the red->green touch sequence")
    ap.add_argument("--headless", action="store_true", help="run without a window")
    ap.add_argument("--no-leds", action="store_true", help="do not draw LED trajectories")
    ap.add_argument("--no-smoothed", action="store_true", help="use raw (unsmoothed) pose columns")
    ap.add_argument("--seed", type=int, default=None, help="random seed for tabletop red/green points")
    ap.add_argument("--num-env", type=int, default=1, help="deprecated; single-environment mode always uses 1")
    ap.add_argument("--env-spacing", type=float, default=1.4, help="deprecated in single-environment mode")
    ap.add_argument("--touch-hold-sec", type=float, default=0.6, help="seconds to hold at each touch target")
    ap.add_argument("--max-ik-steps", type=int, default=260, help="max IK control steps per target")
    ap.add_argument("--touch-pos-tol", type=float, default=0.02, help="position tolerance (m) to accept touch")
    ap.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=0.04,
        help="max per-step joint change (rad) to smooth arm motion",
    )
    ap.add_argument(
        "--traj-duration-sec",
        type=float,
        default=2.0,
        help="planned trajectory duration per move (seconds)",
    )
    ap.add_argument(
        "--output-dir",
        default="data/processed/isaacsim_collection",
        help="output root directory for captured frames/video/trajectory",
    )
    ap.add_argument(
        "--video-filename",
        default="collector_camera.mp4",
        help="base output video file name; per-env videos use env_XX_<name>",
    )
    ap.add_argument(
        "--keep-frames",
        action="store_true",
        help="keep raw PNG frame sequence after mp4 export",
    )
    ap.add_argument(
        "--ik-use-orientation",
        action="store_true",
        help="use corrected-frame orientation in UR5 IK (default tracks corrected position only)",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    trial_tag = "touch_task"
    if args.trial_dir:
        trial_tag = os.path.basename(os.path.abspath(args.trial_dir.rstrip("/"))) or trial_tag
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = os.path.join(args.output_dir, "%s_%s" % (trial_tag, run_stamp))
    frames_dir = os.path.join(run_out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    ee_csv_path = os.path.join(run_out_dir, "ur5_ee_trajectory.csv")
    video_path = os.path.join(run_out_dir, args.video_filename)

    print("[collect-corrected] output dir: %s" % run_out_dir)
    print("[collect-corrected] mode: random tabletop red->green touch (single environment)")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import omni.client
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.extensions import get_extension_path_from_name
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    from isaacsim.core.utils.viewports import set_active_viewport_camera
    from isaacsim.storage.native import get_assets_root_path
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    def enable_camera_light():
        camera_root = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if camera_root and camera_root.IsValid():
            cam_light = UsdLux.DistantLight.Define(stage, "/OmniverseKit_Persp/CameraLight")
            cam_light.CreateIntensityAttr(3500.0)
            cam_light.CreateAngleAttr(1.0)
            return cam_light
        key = UsdLux.DistantLight.Define(stage, "/World/CameraLightFallback")
        key.CreateIntensityAttr(3500.0)
        key.CreateAngleAttr(1.0)
        return key

    enable_camera_light()

    def make_box(path, size_xyz, color, opacity=1.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cube.CreateDisplayOpacityAttr([float(opacity)])
        xf = UsdGeom.Xformable(cube)
        xf.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
        return cube

    def make_axis_triad(root_path, length=0.12, thick=0.01):
        root = UsdGeom.Xform.Define(stage, root_path)
        specs = [
            ("X", (length, thick, thick), (length / 2, 0, 0), (1.0, 0.1, 0.1)),
            ("Y", (thick, length, thick), (0, length / 2, 0), (0.1, 1.0, 0.1)),
            ("Z", (thick, thick, length), (0, 0, length / 2), (0.2, 0.4, 1.0)),
        ]
        for name, size, off, color in specs:
            cube = UsdGeom.Cube.Define(stage, "%s/axis_%s" % (root_path, name))
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(*off))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
        return root

    def make_corrected_frame_marker(root_path="/World/CorrectedFrame", length=0.26, thick=0.008):
        triad = make_axis_triad(root_path, length=length, thick=thick)
        root_prim = stage.GetPrimAtPath(root_path)
        root_prim.CreateAttribute("corrected_frame", Sdf.ValueTypeNames.Bool).Set(True)

        label_specs = [
            ("X", (length + 0.03, 0.0, 0.0), (1.0, 0.1, 0.1)),
            ("Y", (0.0, length + 0.03, 0.0), (0.1, 1.0, 0.1)),
            ("Z", (0.0, 0.0, length + 0.03), (0.2, 0.4, 1.0)),
        ]
        for txt, pos, color in label_specs:
            label = UsdGeom.Xform.Define(stage, "%s/label_%s" % (root_path, txt))
            lxf = UsdGeom.Xformable(label)
            lxf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            lprim = stage.GetPrimAtPath("%s/label_%s" % (root_path, txt))
            lprim.CreateAttribute("debug_label", Sdf.ValueTypeNames.String).Set("%s_c" % txt)
            lprim.CreateAttribute("display_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        return triad

    def make_tabletop(path, size_x, size_y, top_z, thickness=0.03, color=(0.52, 0.46, 0.40)):
        # Center stays on the Z axis (x=0, y=0); top face lies on z=top_z.
        root = UsdGeom.Xform.Define(stage, path)
        root_xf = UsdGeom.Xformable(root)
        root_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(top_z - 0.5 * thickness)))

        cube = UsdGeom.Cube.Define(stage, "%s/surface" % path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        cube.CreateDisplayOpacityAttr([1.0])
        xf = UsdGeom.Xformable(cube)
        xf.AddScaleOp().Set(Gf.Vec3f(float(size_x), float(size_y), float(thickness)))
        return root

    def make_polyline(path, pts, color, width=0.004):
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
        curve.CreateWidthsAttr([float(width)] * len(pts))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return curve

    table_size_x = 1.1
    table_size_y = 0.5
    table_top_z = -1.0
    touch_z = table_top_z + 0.008
    make_corrected_frame_marker("/World/CorrectedFrame", length=0.24, thick=0.007)
    make_tabletop("/World/Table", size_x=table_size_x, size_y=table_size_y, top_z=table_top_z, thickness=0.03)

    margin = 0.08
    x_lo, x_hi = -0.5 * table_size_x + margin, 0.5 * table_size_x - margin
    y_lo, y_hi = -0.5 * table_size_y + margin, 0.5 * table_size_y - margin

    def sample_table_point():
        x = float(rng.uniform(x_lo, x_hi))
        y = float(rng.uniform(y_lo, y_hi))
        return np.array([x, y, touch_z], dtype=np.float64)

    red_target = sample_table_point()
    green_target = sample_table_point()
    for _ in range(40):
        if np.linalg.norm(red_target[:2] - green_target[:2]) >= 0.12:
            break
        green_target = sample_table_point()

    fixed_init_target = np.array([0.0, 0.0, -0.2], dtype=np.float64)
    make_point_marker(stage, "/World/TargetRed", red_target, (1.0, 0.08, 0.08), radius=0.013)
    make_point_marker(stage, "/World/TargetGreen", green_target, (0.1, 1.0, 0.1), radius=0.013)
    make_point_marker(stage, "/World/TargetInit", fixed_init_target, (0.85, 0.85, 0.85), radius=0.011)

    print("[collect-corrected] red target: %s" % red_target)
    print("[collect-corrected] green target: %s" % green_target)
    print("[collect-corrected] fixed init target: %s" % fixed_init_target)

    ur5_ee_log = []

    def create_collection_camera(path, camera_pos_xyz, look_at_xyz):
        cam = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.XformCommonAPI(cam)
        cam_pos = np.asarray(camera_pos_xyz, dtype=np.float64).reshape(3)
        look_at = np.asarray(look_at_xyz, dtype=np.float64).reshape(3)

        # Camera local forward is -Z; use a simple X-tilt model as in the original setup.
        look_dir = look_at - cam_pos
        look_dir /= max(float(np.linalg.norm(look_dir)), 1e-12)
        tilt_x_deg = math.degrees(math.asin(float(look_dir[1])))

        xform.SetTranslate(Gf.Vec3d(*map(float, cam_pos)))
        xform.SetRotate(Gf.Vec3f(float(tilt_x_deg), 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        return path

    collector_camera_path = create_collection_camera(
        "/World/CollectorCamera",
        np.array([0.0, 0.2, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )
    print("[collect-corrected] camera configured at (0.0, 0.2, 0.0), look-at approx (0.0, 0.0, -1.0)")

    global_camera_path = None
    if not args.headless:
        global_cam_pos = np.array([0.0, -1.2, 0.9], dtype=np.float64)
        global_look_at = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        global_camera_path = create_collection_camera("/World/GlobalObserverCamera", global_cam_pos, global_look_at)
        print(
            "[collect-corrected] global observer camera at (%.3f, %.3f, %.3f), look-at (%.3f, %.3f, %.3f)"
            % (
                global_cam_pos[0],
                global_cam_pos[1],
                global_cam_pos[2],
                global_look_at[0],
                global_look_at[1],
                global_look_at[2],
            )
        )

    collector_viewport = None

    def bind_collector_camera():
        nonlocal collector_viewport
        try:
            if (not args.headless) and global_camera_path is not None:
                set_active_viewport_camera(global_camera_path)
            else:
                set_active_viewport_camera(collector_camera_path)
            collector_viewport = get_active_viewport()
        except Exception:
            collector_viewport = None

    bind_collector_camera()
    if collector_viewport is None:
        print("[collect-corrected] warning: no active viewport; video capture disabled")
    else:
        print("[collect-corrected] camera at (0.0, 0.2, 0.0), looking at approx (0.0, 0.0, -1.0)")

    frame_idx = 0

    def capture_frame():
        nonlocal frame_idx
        if collector_viewport is None:
            return

        # Viewport active camera can be reset by world reset/stage changes; force collector camera each capture.
        try:
            set_active_viewport_camera(collector_camera_path)
            out_png = os.path.join(frames_dir, "frame_%06d.png" % frame_idx)
            capture_viewport_to_file(collector_viewport, out_png)
        except Exception:
            pass

        # In GUI mode, restore global observer view after data capture.
        if (not args.headless) and global_camera_path is not None:
            try:
                set_active_viewport_camera(global_camera_path)
            except Exception:
                pass
        frame_idx += 1

    def resolve_ur5_usd_path():
        assets_root = get_assets_root_path()
        if not assets_root:
            return None

        candidates = [
            "/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
            "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        ]

        for rel in candidates:
            url = assets_root + rel
            try:
                status, _entry = omni.client.stat(url)
                if status == omni.client.Result.OK:
                    return url
            except Exception:
                continue

        # Keep a best-effort fallback when stat() is unavailable in some deployments.
        return assets_root + candidates[0]

    def build_ur5_ik_solver(robot_articulation):
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        if not mg_extension_path:
            raise RuntimeError("cannot locate extension path: isaacsim.robot_motion.motion_generation")

        urdf_path = os.path.join(mg_extension_path, "motion_policy_configs/universal_robots/ur5/ur5.urdf")
        robot_desc_yaml = os.path.join(
            mg_extension_path,
            "motion_policy_configs/universal_robots/ur5/rmpflow/ur5_robot_description.yaml",
        )

        kin = LulaKinematicsSolver(robot_description_path=robot_desc_yaml, urdf_path=urdf_path)
        frames = list(kin.get_all_frame_names())
        if not frames:
            raise RuntimeError("UR5 kinematics returned no frame names")

        preferred = [
            "tool0",
            "flange",
            "ee_link",
            "wrist_3_link",
            "wrist_3",
            "tcp",
        ]

        ee_frame = None
        for name in preferred:
            if name in frames:
                ee_frame = name
                break

        if ee_frame is None:
            for name in frames:
                low = str(name).lower()
                if "tool" in low or "flange" in low or "tcp" in low:
                    ee_frame = name
                    break

        if ee_frame is None:
            ee_frame = frames[-1]

        print("[replay-corrected] UR5 IK ee frame: %s" % ee_frame)
        return ArticulationKinematicsSolver(robot_articulation, kin, ee_frame)

    def sync_ur5_base_pose(ik_solver, base_pos, base_quat_wxyz):
        if ik_solver is None:
            return
        kin = ik_solver.get_kinematics_solver()
        kin.set_robot_base_pose(np.asarray(base_pos, dtype=np.float64), np.asarray(base_quat_wxyz, dtype=np.float64))

    world = World(stage_units_in_meters=1.0)
    ur5_ctrl = None
    ur5_ik = None
    ur5_joint_subset = None
    ur5_safe_frames = []
    ur5_base_pos = np.array([0.0, 0.5, -1.0], dtype=np.float64)
    ur5_base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ur5_min_link_z = -1.0 + 1e-4

    def _quat_wxyz_from_yaw(yaw_rad):
        half = 0.5 * float(yaw_rad)
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)

    def choose_base_quat_from_first_ik(ur5_robot, ik_solver, target_pos, target_quat):
        best_quat = None
        best_score = float("inf")
        # Search base yaw on Z; keep roll/pitch as 0 for a stable mounted base.
        for yaw in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            q = _quat_wxyz_from_yaw(yaw)
            ur5_robot.set_world_pose(position=ur5_base_pos, orientation=q)
            sync_ur5_base_pose(ik_solver, ur5_base_pos, q)
            world.step(render=False)
            try:
                actions, success = ik_solver.compute_inverse_kinematics(
                    target_position=np.asarray(target_pos, dtype=np.float64),
                    target_orientation=(
                        np.asarray(target_quat, dtype=np.float64) if args.ik_use_orientation else None
                    ),
                )
            except Exception:
                continue
            if not success:
                continue

            score = abs(float(yaw))
            joint_pos = getattr(actions, "joint_positions", None)
            if joint_pos is not None:
                arr = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
                if arr.size > 0 and np.isfinite(arr).all():
                    score = float(np.linalg.norm(arr))

            if score < best_score:
                best_score = score
                best_quat = q

        if best_quat is None:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return best_quat

    def choose_ur5_safe_frames(ik_solver):
        kin = ik_solver.get_kinematics_solver()
        frames = list(kin.get_all_frame_names())
        preferred = [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "tool0",
        ]
        picked = [name for name in preferred if name in frames]
        if picked:
            return picked
        return frames[-min(6, len(frames)):]

    def joints_keep_robot_above_table(ik_solver, safe_frames, joint_positions):
        kin = ik_solver.get_kinematics_solver()
        arr = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        min_z = float("inf")
        for frame_name in safe_frames:
            try:
                frame_pos, _frame_rot = kin.compute_forward_kinematics(frame_name, arr, position_only=True)
            except Exception:
                return False, float("-inf")
            frame_pos = np.asarray(frame_pos, dtype=np.float64).reshape(3)
            if not np.isfinite(frame_pos).all():
                return False, float("-inf")
            min_z = min(min_z, float(frame_pos[2]))
        return min_z > ur5_min_link_z, min_z

    def compute_safe_ik_action(ik_solver, joint_subset, safe_frames, target_pos, target_quat):
        kin = ik_solver.get_kinematics_solver()
        current = joint_subset.get_joint_positions()
        warm_starts = []
        if current is not None:
            current = np.asarray(current, dtype=np.float64).reshape(-1)
            warm_starts.append(current)

        canonical_seeds = [
            np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64),
            np.array([0.0, -1.20, 1.80, -1.90, -1.57, 0.0], dtype=np.float64),
            np.array([0.0, -0.90, 1.20, -1.80, -1.57, 0.0], dtype=np.float64),
            np.array([math.pi, -1.20, 1.80, -1.90, 1.57, 0.0], dtype=np.float64),
            np.array([-math.pi, -1.20, 1.80, -1.90, 1.57, 0.0], dtype=np.float64),
            np.zeros(6, dtype=np.float64),
        ]
        for seed in canonical_seeds:
            if current is not None and seed.shape == current.shape and np.allclose(seed, current):
                continue
            warm_starts.append(seed)

        best = None
        best_score = None
        ee_frame = ik_solver.get_end_effector_frame()
        for warm in warm_starts:
            try:
                joint_sol, success = kin.compute_inverse_kinematics(
                    ee_frame,
                    np.asarray(target_pos, dtype=np.float64),
                    (np.asarray(target_quat, dtype=np.float64) if target_quat is not None else None),
                    warm_start=np.asarray(warm, dtype=np.float64),
                )
            except Exception:
                continue
            if not success:
                continue
            joint_sol = np.asarray(joint_sol, dtype=np.float64).reshape(-1)
            safe, min_z = joints_keep_robot_above_table(ik_solver, safe_frames, joint_sol)
            if not safe:
                continue
            score = -10.0 * min_z
            if current is not None and current.shape == joint_sol.shape:
                score += float(np.linalg.norm(joint_sol - current))
            else:
                score += 0.1 * float(np.linalg.norm(joint_sol))
            if best is None or score < best_score:
                best = joint_sol
                best_score = score

        if best is None:
            return None, False
        return joint_subset.make_articulation_action(best, None), True

    def compute_relaxed_ik_action(ik_solver, target_pos, target_quat):
        try:
            return ik_solver.compute_inverse_kinematics(
                target_position=np.asarray(target_pos, dtype=np.float64),
                target_orientation=(np.asarray(target_quat, dtype=np.float64) if target_quat is not None else None),
            )
        except Exception:
            return None, False

    def apply_smoothed_action(ctrl, joint_subset, actions):
        if actions is None or ctrl is None or joint_subset is None:
            return None
        target = getattr(actions, "joint_positions", None)
        if target is None:
            ctrl.apply_action(actions)
            return None

        current = joint_subset.get_joint_positions()
        if current is None:
            ctrl.apply_action(actions)
            return None

        target = np.asarray(target, dtype=np.float64).reshape(-1)
        current = np.asarray(current, dtype=np.float64).reshape(-1)
        if target.shape != current.shape:
            ctrl.apply_action(actions)
            return None

        max_step = max(1e-4, float(args.max_joint_step_rad))
        delta = np.clip(target - current, -max_step, max_step)
        smoothed = current + delta
        ctrl.apply_action(joint_subset.make_articulation_action(smoothed, None))
        return smoothed

    def compute_tracking_ik_action(ik_solver, joint_subset, target_pos, target_quat, warm_start_joints):
        kin = ik_solver.get_kinematics_solver()
        ee_frame = ik_solver.get_end_effector_frame()

        if warm_start_joints is None:
            warm_start_joints = joint_subset.get_joint_positions()
        if warm_start_joints is None:
            return None, False

        warm_start_joints = np.asarray(warm_start_joints, dtype=np.float64).reshape(-1)
        try:
            joint_sol, success = kin.compute_inverse_kinematics(
                ee_frame,
                np.asarray(target_pos, dtype=np.float64),
                (np.asarray(target_quat, dtype=np.float64) if target_quat is not None else None),
                warm_start=warm_start_joints,
            )
        except Exception:
            return None, False

        if not success:
            return None, False
        joint_sol = np.asarray(joint_sol, dtype=np.float64).reshape(-1)
        return joint_subset.make_articulation_action(joint_sol, None), True

    def hermite_segment(p0, p1, v0, v1, dt, t_local):
        if dt <= 1e-9:
            return np.asarray(p1, dtype=np.float64)
        u = float(np.clip(t_local / dt, 0.0, 1.0))
        h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0
        h10 = u * u * u - 2.0 * u * u + u
        h01 = -2.0 * u * u * u + 3.0 * u * u
        h11 = u * u * u - u * u
        return h00 * p0 + h10 * dt * v0 + h01 * p1 + h11 * dt * v1

    def smoothstep01(u):
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    state = None
    try:
        ur5_usd_path = resolve_ur5_usd_path()
        if not ur5_usd_path:
            raise RuntimeError("UR5 asset root unavailable")

        ur5_prim_path = "/World/UR5"
        add_reference_to_stage(usd_path=ur5_usd_path, prim_path=ur5_prim_path)
        robot = world.scene.add(
            Robot(
                prim_path=ur5_prim_path,
                name="ur5",
                position=np.array([0.0, 0.5, -1.0], dtype=np.float64),
            )
        )
        state = {
            "robot": robot,
            "fixed_target": np.asarray(fixed_init_target, dtype=np.float64),
            "red_target": np.asarray(red_target, dtype=np.float64),
            "green_target": np.asarray(green_target, dtype=np.float64),
        }

        world.reset()
        bind_collector_camera()

        ctrl = robot.get_articulation_controller()
        ik = build_ur5_ik_solver(robot)
        joint_subset = ik.get_joints_subset()
        safe_frames = choose_ur5_safe_frames(ik)

        base_pos = np.array([0.0, 0.5, -1.0], dtype=np.float64)
        base_quat = choose_base_quat_from_first_ik(robot, ik, state["red_target"], None)
        robot.set_world_pose(position=base_pos, orientation=base_quat)
        robot.set_default_state(position=base_pos, orientation=base_quat)
        sync_ur5_base_pose(ik, base_pos, base_quat)

        state.update(
            {
                "ctrl": ctrl,
                "ik": ik,
                "joint_subset": joint_subset,
                "safe_frames": safe_frames,
                "phase": "fixed_init",
                "phase_index": 0,
                "q_start": None,
                "q_goal": None,
                "plan_step": 0,
                "plan_steps": 0,
                "hold_steps_left": 0,
                "done": False,
            }
        )
        print("[collect-corrected] UR5 ready, safe-frames=%s" % (",".join(safe_frames)))

    except Exception as ex:
        print("[replay-corrected] warning: UR5 IK integration unavailable: %s" % str(ex))
        world.reset()
        bind_collector_camera()
        state = None

    run_start_wall = time.time()

    def get_target_for_phase(st):
        phase = st["phase"]
        if phase == "fixed_init":
            return st["fixed_target"]
        if phase == "initial_to_red":
            return st["red_target"]
        return st["green_target"]

    def log_ee_all(now_t):
        if state is None:
            return
        ik = state.get("ik")
        if ik is None:
            return
        try:
            ee_pos, _ee_rot = ik.compute_end_effector_pose(position_only=True)
            ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
            if not np.isfinite(ee_pos).all():
                return
            ur5_ee_log.append(
                (
                    str(state.get("phase", "unknown")),
                    float(now_t),
                    float(ee_pos[0]),
                    float(ee_pos[1]),
                    float(ee_pos[2]),
                )
            )
        except Exception:
            return

    def next_phase(st):
        phase = st["phase"]
        if phase == "fixed_init":
            st["phase"] = "initial_to_red"
        elif phase == "initial_to_red":
            st["phase"] = "green"
        elif phase == "green":
            if args.loop:
                st["phase"] = "initial_to_red"
            else:
                st["done"] = True
        st["q_start"] = None
        st["q_goal"] = None
        st["plan_step"] = 0
        st["plan_steps"] = 0
        st["hold_steps_left"] = 0

    def init_phase_plan(st):
        if st.get("done"):
            return
        joint_subset = st["joint_subset"]
        ik = st["ik"]
        safe_frames = st["safe_frames"]
        q_start = joint_subset.get_joint_positions()
        if q_start is None:
            return
        q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
        target_pos = get_target_for_phase(st)

        actions, success = compute_tracking_ik_action(ik, joint_subset, target_pos, None, q_start)
        if (not success) or (actions is None):
            actions, success = compute_safe_ik_action(ik, joint_subset, safe_frames, target_pos, None)
        if (not success) or (actions is None):
            actions, success = compute_relaxed_ik_action(ik, target_pos, None)
        if (not success) or (actions is None):
            print("[collect-corrected] warning: phase %s IK failed" % (st["phase"]))
            return

        q_goal = getattr(actions, "joint_positions", None)
        if q_goal is None:
            return
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(-1)
        if q_goal.shape != q_start.shape or (not np.isfinite(q_goal).all()):
            return

        total_dur = max(0.2, float(args.traj_duration_sec))
        fps_eff = float(args.fps if args.fps and args.fps > 0 else 60.0)
        st["q_start"] = q_start
        st["q_goal"] = q_goal
        st["plan_step"] = 0
        st["plan_steps"] = max(4, int(math.ceil(total_dur * fps_eff)))
        st["hold_steps_left"] = max(1, int(float(args.touch_hold_sec) * fps_eff))

    def update_env_one_step(st):
        if st.get("done"):
            return
        if st.get("q_goal") is None:
            init_phase_plan(st)
            if st.get("q_goal") is None:
                return

        ctrl = st["ctrl"]
        joint_subset = st["joint_subset"]
        q_start = st["q_start"]
        q_goal = st["q_goal"]

        if st["plan_step"] < st["plan_steps"]:
            u = float(st["plan_step"]) / float(max(1, st["plan_steps"] - 1))
            s = smoothstep01(u)
            q_ref = q_start + s * (q_goal - q_start)
            q_action = joint_subset.make_articulation_action(q_ref, None)
            apply_smoothed_action(ctrl, joint_subset, q_action)
            st["plan_step"] += 1
            return

        if st["hold_steps_left"] > 0:
            q_action = joint_subset.make_articulation_action(q_goal, None)
            apply_smoothed_action(ctrl, joint_subset, q_action)
            st["hold_steps_left"] -= 1
            if st["hold_steps_left"] <= 0:
                try:
                    ee_pos, _ee_rot = st["ik"].compute_end_effector_pose(position_only=True)
                    ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
                    reached = bool(
                        np.isfinite(ee_pos).all()
                        and np.linalg.norm(ee_pos - get_target_for_phase(st)) <= float(args.touch_pos_tol)
                    )
                except Exception:
                    reached = False
                print("[collect-corrected] target %s reached=%s" % (st["phase"], reached))
                next_phase(st)

    for _ in range(5):
        world.step(render=True)
        capture_frame()

    print("[collect-corrected] touch sequence start (single environment)")

    frame_dt = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    next_frame = time.time()
    while simulation_app.is_running():
        if (state is not None) and bool(state.get("done")) and (not args.loop):
            break

        if state is not None:
            update_env_one_step(state)

        world.step(render=True)
        capture_frame()
        now_t = time.time() - run_start_wall
        log_ee_all(now_t)

        if frame_dt > 0.0:
            next_frame += frame_dt
            sleep_t = next_frame - time.time()
            if sleep_t > 0.0:
                time.sleep(sleep_t)
            else:
                next_frame = time.time()

    with open(ee_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "t_sec", "x_m", "y_m", "z_m"])
        writer.writerows(ur5_ee_log)
    print("[collect-corrected] ee trajectory saved: %s (%d rows)" % (ee_csv_path, len(ur5_ee_log)))

    if frame_idx > 0:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            video_fps = args.fps if args.fps and args.fps > 0 else 60.0
            frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
            ffmpeg_cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                "%.6f" % float(video_fps),
                "-i",
                frame_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
            proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                print("[collect-corrected] video saved: %s" % video_path)
            else:
                print("[collect-corrected] warning: ffmpeg failed, keep PNG frames in %s" % frames_dir)
                if proc.stderr:
                    err_lines = proc.stderr.strip().splitlines()
                    if err_lines:
                        print("[collect-corrected] ffmpeg stderr tail: %s" % err_lines[-1])

            if proc.returncode == 0 and not args.keep_frames:
                shutil.rmtree(frames_dir, ignore_errors=True)
                print("[collect-corrected] raw frames removed: %s" % frames_dir)
        else:
            print("[collect-corrected] warning: ffmpeg not found, keep PNG frames in %s" % frames_dir)
    else:
        print("[collect-corrected] warning: no frames captured")

    simulation_app.close()


if __name__ == "__main__":
    main()
