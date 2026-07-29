#!/usr/bin/env python3
"""Isaac Sim kinematic replay in corrected-frame world coordinates.

This script replays PRISM rigid 6D trajectories just like tools/isaacsim_replay.py,
but remaps data into the corrected frame defined by camera centers + calibration.
As a result, Isaac Sim's world frame matches the online preview's "Corrected Frame".

Usage:
    /isaac-sim/python.sh tools/isaacsim_replay_corrected.py \
        --trial-dir data/raw/task_xxx/trial_000001 \
        --calib-json configs/devices/charuco_4cam_result.json
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

# Make sure "src/" is importable when running from repository root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from prism.reconstruction.calibration import (  # noqa: E402
    apply_corrected_transform,
    build_corrected_transform,
    get_camera_centers_world,
    load_calibration,
)

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial-dir", help="trial dir containing trajectory/ and hand/")
    ap.add_argument("--rigid", help="path to rigid_pose_6d.csv")
    ap.add_argument("--led", help="path to trajectory_led.csv (long format)")
    ap.add_argument("--gestures", help="path to sdk_commands.csv / gesture timeline CSV")
    ap.add_argument("--calib-json", required=True, help="charuco calibration json used for corrected frame")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--fps", type=float, default=60.0, help="render frame-rate cap (0 = uncapped)")
    ap.add_argument("--loop", action="store_true", help="loop playback")
    ap.add_argument("--headless", action="store_true", help="run without a window")
    ap.add_argument("--no-leds", action="store_true", help="do not draw LED trajectories")
    ap.add_argument("--no-smoothed", action="store_true", help="use raw (unsmoothed) pose columns")
    ap.add_argument(
        "--ik-use-orientation",
        action="store_true",
        help="use corrected-frame orientation in UR5 IK (default tracks corrected position only)",
    )
    args = ap.parse_args()

    rigid_path, led_path, gest_path = resolve_inputs(args)
    t_arr, pos_arr, rot_arr, used_sm = load_poses(rigid_path, prefer_smoothed=not args.no_smoothed)
    pos_arr_world = np.asarray(pos_arr, dtype=np.float64).copy()
    leds = {} if args.no_leds else load_leds(led_path)
    gestures = load_gestures(gest_path)

    pos_arr, rot_arr, leds = transform_to_corrected_frame(pos_arr, rot_arr, leds, args.calib_json)
    quat_arr = np.asarray([_mat_to_quat(r) for r in rot_arr], dtype=np.float64)

    dur = float(t_arr[-1] - t_arr[0])
    print("[replay-corrected] poses: %d samples, %.2fs, smoothed=%s" % (len(t_arr), dur, str(used_sm)))
    print("[replay-corrected] LED colors: %s" % (sorted(leds.keys()) or "none"))
    print("[replay-corrected] gesture events: %d" % len(gestures))
    print(
        "[replay-corrected] rigid bounds world min/max: %s / %s"
        % (np.min(pos_arr_world, axis=0), np.max(pos_arr_world, axis=0))
    )
    print(
        "[replay-corrected] rigid bounds corrected min/max: %s / %s"
        % (np.min(pos_arr, axis=0), np.max(pos_arr, axis=0))
    )
    print("[replay-corrected] first rigid world->corrected: %s -> %s" % (pos_arr_world[0], pos_arr[0]))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import omni.client
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.extensions import get_extension_path_from_name
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
    from isaacsim.storage.native import get_assets_root_path
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

    make_corrected_frame_marker("/World/CorrectedFrame", length=0.24, thick=0.007)
    make_tabletop("/World/Table", size_x=1.1, size_y=0.5, top_z=-1.0, thickness=0.03)
    rigid_curve = make_polyline("/World/RigidReplayTrajectory", pos_arr, (1.0, 0.3, 0.8), width=0.0035)
    ur5_ee_curve = make_polyline("/World/UR5EETrajectory", [pos_arr[0]], (0.1, 1.0, 1.0), width=0.0055)
    ur5_ee_hist = []

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

    def joints_keep_robot_above_table(ik_solver, joint_positions):
        kin = ik_solver.get_kinematics_solver()
        arr = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        min_z = float("inf")
        for frame_name in ur5_safe_frames:
            try:
                frame_pos, _frame_rot = kin.compute_forward_kinematics(frame_name, arr, position_only=True)
            except Exception:
                return False, float("-inf")
            frame_pos = np.asarray(frame_pos, dtype=np.float64).reshape(3)
            if not np.isfinite(frame_pos).all():
                return False, float("-inf")
            min_z = min(min_z, float(frame_pos[2]))
        return min_z > ur5_min_link_z, min_z

    def compute_safe_ik_action(ik_solver, target_pos, target_quat):
        kin = ik_solver.get_kinematics_solver()
        joint_subset = ur5_joint_subset
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
            safe, min_z = joints_keep_robot_above_table(ik_solver, joint_sol)
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

    try:
        ur5_prim_path = "/World/UR5"
        ur5_usd_path = resolve_ur5_usd_path()
        if not ur5_usd_path:
            raise RuntimeError("UR5 asset root unavailable")

        add_reference_to_stage(usd_path=ur5_usd_path, prim_path=ur5_prim_path)
        ur5 = world.scene.add(
            Robot(
                prim_path=ur5_prim_path,
                name="ur5",
                position=ur5_base_pos,
            )
        )
        world.reset()
        ur5_ctrl = ur5.get_articulation_controller()
        ur5_ik = build_ur5_ik_solver(ur5)
        ur5_joint_subset = ur5_ik.get_joints_subset()
        ur5_safe_frames = choose_ur5_safe_frames(ur5_ik)

        # Solve base yaw from the first replay target instead of hardcoding wxyz=(1,0,0,0).
        ur5_base_quat_wxyz = choose_base_quat_from_first_ik(ur5, ur5_ik, pos_arr[0], quat_arr[0])

        # Reset may restore articulation to USD defaults; explicitly pin base pose.
        ur5.set_world_pose(position=ur5_base_pos, orientation=ur5_base_quat_wxyz)
        ur5.set_default_state(position=ur5_base_pos, orientation=ur5_base_quat_wxyz)
        sync_ur5_base_pose(ur5_ik, ur5_base_pos, ur5_base_quat_wxyz)

        # Initialize arm joints from IK at the first replay pose so playback starts aligned.
        first_actions, first_ok = compute_safe_ik_action(
            ur5_ik,
            np.asarray(pos_arr[0], dtype=np.float64),
            (np.asarray(quat_arr[0], dtype=np.float64) if args.ik_use_orientation else None),
        )
        if first_ok:
            ur5_ctrl.apply_action(first_actions)

        print("[replay-corrected] UR5 loaded: %s" % ur5_usd_path)
        print("[replay-corrected] UR5 base aligned at (0.0, 0.5, -1.0)")
        print(
            "[replay-corrected] UR5 base wxyz from first-point IK: (%.6f, %.6f, %.6f, %.6f)"
            % (
                ur5_base_quat_wxyz[0],
                ur5_base_quat_wxyz[1],
                ur5_base_quat_wxyz[2],
                ur5_base_quat_wxyz[3],
            )
        )
        print("[replay-corrected] UR5 Lula base pose synchronized with articulation pose")
        print("[replay-corrected] UR5 safe-link frames: %s" % ", ".join(ur5_safe_frames))
    except Exception as ex:
        print("[replay-corrected] warning: UR5 IK integration unavailable: %s" % str(ex))
        # Keep replay alive even if UR5 cannot be loaded on this host.
        world.reset()

    hand_xform = UsdGeom.Xform.Define(stage, "/World/Hand")
    hand_op = hand_xform.MakeMatrixXform()
    make_box("/World/Hand/palm", (0.09, 0.06, 0.02), (0.6, 0.6, 0.65), opacity=0.55)
    make_axis_triad("/World/Hand/body_frame", length=0.1, thick=0.008)

    for color, pts in leds.items():
        if len(pts) < 2:
            continue
        curve = UsdGeom.BasisCurves.Define(stage, "/World/LED_%s" % color)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
        curve.CreateWidthsAttr([0.004] * len(pts))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        rgb = LED_COLORS.get(color, (0.8, 0.8, 0.8))
        curve.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def set_hand(pos, quat):
        r = _quat_to_mat(quat)
        rt = r.T
        m = Gf.Matrix4d(
            rt[0, 0], rt[0, 1], rt[0, 2], 0.0,
            rt[1, 0], rt[1, 1], rt[1, 2], 0.0,
            rt[2, 0], rt[2, 1], rt[2, 2], 0.0,
            float(pos[0]), float(pos[1]), float(pos[2]), 1.0,
        )
        hand_op.Set(m)

    set_hand(pos_arr[0], quat_arr[0])
    for _ in range(5):
        world.step(render=True)

    if ur5_ik is not None:
        try:
            ee_pos0, _ee_rot0 = ur5_ik.compute_end_effector_pose(position_only=True)
            ee_pos0 = np.asarray(ee_pos0, dtype=np.float64).reshape(3)
            if np.isfinite(ee_pos0).all():
                ur5_ee_hist.append(ee_pos0.copy())
                ur5_ee_curve.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in ur5_ee_hist])
                ur5_ee_curve.GetCurveVertexCountsAttr().Set([len(ur5_ee_hist)])
                ur5_ee_curve.GetWidthsAttr().Set([0.0055] * len(ur5_ee_hist))
        except Exception:
            pass

    t0 = float(t_arr[0])
    frame_dt = 1.0 / args.fps if args.fps and args.fps > 0 else 0.0
    print("[replay-corrected] playing... (close window or Ctrl-C to stop)")
    while simulation_app.is_running():
        start_wall = time.time()
        next_frame = time.time()
        g_idx = 0
        while simulation_app.is_running():
            now = t0 + (time.time() - start_wall) * args.speed
            pos, quat = sample_pose(t_arr, pos_arr, quat_arr, now)
            set_hand(pos, quat)

            if ur5_ctrl is not None and ur5_ik is not None:
                try:
                    actions, success = compute_safe_ik_action(
                        ur5_ik,
                        np.asarray(pos, dtype=np.float64),
                        (np.asarray(quat, dtype=np.float64) if args.ik_use_orientation else None),
                    )
                    if success:
                        ur5_ctrl.apply_action(actions)
                except Exception:
                    # Ignore transient IK failures for individual frames.
                    pass

            while g_idx < len(gestures) and gestures[g_idx][0] <= now:
                gt, label = gestures[g_idx]
                print("[gesture] t=%.3fs  %s" % (gt, label))
                g_idx += 1
            world.step(render=True)

            if ur5_ik is not None:
                try:
                    ee_pos, _ee_rot = ur5_ik.compute_end_effector_pose(position_only=True)
                    ee_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3)
                    if np.isfinite(ee_pos).all():
                        if (not ur5_ee_hist) or np.linalg.norm(ee_pos - ur5_ee_hist[-1]) > 1e-6:
                            ur5_ee_hist.append(ee_pos.copy())
                            ur5_ee_curve.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in ur5_ee_hist])
                            ur5_ee_curve.GetCurveVertexCountsAttr().Set([len(ur5_ee_hist)])
                            ur5_ee_curve.GetWidthsAttr().Set([0.0055] * len(ur5_ee_hist))
                except Exception:
                    pass

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
