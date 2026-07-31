#!/usr/bin/env python3
"""Replay planned AUBO i5 + MechHand joint targets in Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import yaml


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()


def load_robot_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_plan(path: Path, joint_names: Sequence[str], max_frames: int = 0, stride: int = 1) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = set(reader.fieldnames or [])
        missing = [name for name in joint_names if name not in cols]
        if missing:
            raise SystemExit("planned motion missing joint columns: %s" % ", ".join(missing))
        for index, row in enumerate(reader):
            if stride > 1 and index % stride != 0:
                continue
            item = {
                "t_sec": float(row.get("t_sec", "0") or 0.0),
                "joints": {name: float(row[name]) for name in joint_names},
                "ik_success": int(float(row.get("ik_success", "1") or 1)),
                "hand_pose_name": row.get("hand_pose_name", ""),
                "target_hand": np.array(
                    [
                        float(row.get("target_hand_x_m", "nan")),
                        float(row.get("target_hand_y_m", "nan")),
                        float(row.get("target_hand_z_m", "nan")),
                    ],
                    dtype=np.float64,
                ),
            }
            rows.append(item)
            if max_frames > 0 and len(rows) >= max_frames:
                break
    if not rows:
        raise SystemExit("no planned motion rows found: %s" % path)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planned-motion", required=True, help="planned_motion.csv from plan_robot_motion.py")
    parser.add_argument(
        "--robot-config",
        default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"),
        help="robot asset and placement YAML",
    )
    parser.add_argument(
        "--out-usd",
        default=str(_REPO_ROOT / "simulation" / "scenes" / "aubo_i5_mechhand_replay.usd"),
        help="output USD after replaying the requested frames",
    )
    parser.add_argument("--prim-path", default=None, help="override robot prim path")
    parser.add_argument("--headless", action="store_true", help="run without viewport")
    parser.add_argument("--no-preview", action="store_true", help="exit after replay instead of holding the window open")
    parser.add_argument("--max-frames", type=int, default=0, help="replay only the first N selected frames")
    parser.add_argument("--stride", type=int, default=1, help="replay every Nth planned row")
    parser.add_argument("--speed", type=float, default=1.0, help="wall-clock playback speed for GUI replay")
    parser.add_argument("--fast", action="store_true", help="do not sleep between frames, even with a visible viewport")
    parser.add_argument("--render-every", type=int, default=1, help="render every Nth frame; headless still steps physics")
    parser.add_argument("--record-video", action="store_true", help="record the replay from a fixed overhead camera")
    parser.add_argument("--video-path", default=None, help="mp4 output path; defaults next to planned_motion.csv")
    parser.add_argument("--frames-dir", default=None, help="raw PNG frame directory for video recording")
    parser.add_argument("--keep-frames", action="store_true", help="keep raw PNG frames after successful mp4 export")
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.2, 0.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, -0.9], metavar=("X", "Y", "Z"))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.robot_config).expanduser().resolve()
    config = load_robot_config(config_path)
    joint_names = list(config.get("arm_joints", [])) + list(config.get("hand_joints", []))
    if not joint_names:
        raise SystemExit("robot config does not define arm_joints/hand_joints: %s" % config_path)

    planned_motion = Path(args.planned_motion).expanduser().resolve()
    rows = load_plan(planned_motion, joint_names, max_frames=max(0, int(args.max_frames)), stride=max(1, int(args.stride)))
    robot_usd = resolve_repo_path(config.get("combined_usd", ""))
    if not robot_usd.is_file():
        raise SystemExit("robot USD not found: %s" % robot_usd)
    out_usd = Path(args.out_usd).expanduser().resolve()
    video_path = Path(args.video_path).expanduser().resolve() if args.video_path else planned_motion.with_name("planned_motion_overhead.mp4")
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else planned_motion.parent / "planned_motion_overhead_frames"

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_active_viewport_camera
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    usd_context = omni.usd.get_context()
    usd_context.open_stage(str(robot_usd))
    for _ in range(4):
        simulation_app.update()
    stage = usd_context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot_prim_path = str(args.prim_path or config.get("robot_prim", "/World/AUBO_i5_MechHand"))
    if not stage.GetPrimAtPath(robot_prim_path).IsValid() and stage.GetDefaultPrim():
        robot_prim_path = str(stage.GetDefaultPrim().GetPath())

    def make_box(path: str, size_xyz: Sequence[float], color: Sequence[float], opacity: float = 1.0):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
        cube.CreateDisplayOpacityAttr([float(opacity)])
        UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size_xyz]))
        return cube

    def make_polyline(path: str, points: Iterable[Sequence[float]], color: Sequence[float], width: float = 0.004):
        pts = [Gf.Vec3f(*[float(v) for v in point]) for point in points]
        if len(pts) < 2:
            return None
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(pts)])
        curve.CreatePointsAttr(pts)
        curve.CreateWidthsAttr([float(width)] * len(pts))
        curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        curve.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in color])])
        return curve

    def make_axis_triad(root_path: str, length: float = 0.25, thick: float = 0.008):
        UsdGeom.Xform.Define(stage, root_path)
        for name, size, off, color in [
            ("X", (length, thick, thick), (length / 2, 0.0, 0.0), (1.0, 0.1, 0.1)),
            ("Y", (thick, length, thick), (0.0, length / 2, 0.0), (0.1, 1.0, 0.1)),
            ("Z", (thick, thick, length), (0.0, 0.0, length / 2), (0.2, 0.4, 1.0)),
        ]:
            cube = UsdGeom.Cube.Define(stage, "%s/axis_%s" % (root_path, name))
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xf = UsdGeom.Xformable(cube)
            xf.AddTranslateOp().Set(Gf.Vec3d(*off))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))

    def find_articulation_root(root_path: str) -> str:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError("robot prim does not exist: %s" % root_path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
        return root_path

    def create_overhead_camera(path: str, camera_pos_xyz: Sequence[float], look_at_xyz: Sequence[float]) -> str:
        cam = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.XformCommonAPI(cam)
        cam_pos = np.asarray(camera_pos_xyz, dtype=np.float64).reshape(3)
        look_at = np.asarray(look_at_xyz, dtype=np.float64).reshape(3)
        look_dir = look_at - cam_pos
        look_dir /= max(float(np.linalg.norm(look_dir)), 1e-12)
        tilt_x_deg = math.degrees(math.asin(float(look_dir[1])))
        xform.SetTranslate(Gf.Vec3d(*[float(v) for v in cam_pos]))
        xform.SetRotate(Gf.Vec3f(float(tilt_x_deg), 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        return path

    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2800.0)
    key.CreateAngleAttr(1.0)
    fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(450.0)
    make_axis_triad("/World/CorrectedFrame", length=0.28, thick=0.008)
    UsdGeom.Xform.Define(stage, "/World/Table")
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Table")).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -1.015))
    make_box("/World/Table/surface", (1.2, 0.7, 0.03), (0.50, 0.45, 0.38), opacity=1.0)

    target_marker = make_box("/World/ReplayTarget", (0.018, 0.018, 0.018), (1.0, 0.9, 0.1), opacity=1.0)
    target_marker_xform = UsdGeom.Xformable(target_marker)
    target_marker_translate = target_marker_xform.AddTranslateOp()
    overhead_camera_path = create_overhead_camera("/World/OverheadReplayCamera", args.camera_pos, args.camera_look_at)
    print(
        "[planned-replay] overhead camera: pos=(%.3f, %.3f, %.3f), look-at=(%.3f, %.3f, %.3f)"
        % (*[float(v) for v in args.camera_pos], *[float(v) for v in args.camera_look_at])
    )

    collector_viewport = None
    frame_index_for_video = 0
    if args.record_video:
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            set_active_viewport_camera(overhead_camera_path)
            collector_viewport = get_active_viewport()
        except Exception as exc:
            collector_viewport = None
            print("[planned-replay] WARN: video capture disabled, no active viewport: %s" % exc)
        if collector_viewport is not None:
            print("[planned-replay] recording video frames to: %s" % frames_dir)

    def capture_frame() -> None:
        nonlocal frame_index_for_video
        if collector_viewport is None:
            return
        try:
            set_active_viewport_camera(overhead_camera_path)
            out_png = frames_dir / ("frame_%06d.png" % frame_index_for_video)
            capture_viewport_to_file(collector_viewport, str(out_png))
            frame_index_for_video += 1
        except Exception as exc:
            print("[planned-replay] WARN: frame capture failed: %s" % exc)

    world_from_base = config.get("world_from_base", {}) or {}
    base_xyz = np.asarray(world_from_base.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    base_quat = np.asarray(world_from_base.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)

    world = World(stage_units_in_meters=1.0)
    world.reset()
    omni.timeline.get_timeline_interface().play()
    for _ in range(8):
        simulation_app.update()

    articulation_root_path = find_articulation_root(robot_prim_path)
    articulation = SingleArticulation(prim_path=articulation_root_path)
    articulation.initialize()
    articulation.set_world_pose(position=base_xyz, orientation=base_quat)
    dof_names = list(articulation.dof_names)
    dof_index = {name: index for index, name in enumerate(dof_names)}
    missing_dofs = [name for name in joint_names if name not in dof_index]
    if missing_dofs:
        raise SystemExit("robot missing planned DOFs: %s" % ", ".join(missing_dofs))

    print("[planned-replay] opened robot USD: %s" % robot_usd)
    print("[planned-replay] articulation root: %s" % articulation_root_path)
    print("[planned-replay] DOF count: %d" % len(dof_names))
    print("[planned-replay] frames: %d" % len(rows))
    print("[planned-replay] IK-success rows: %d/%d" % (sum(row["ik_success"] for row in rows), len(rows)))

    joint_positions = np.zeros(len(dof_names), dtype=np.float64)
    zero_velocities = np.zeros(len(dof_names), dtype=np.float64)
    render_every = max(1, int(args.render_every))
    previous_t = rows[0]["t_sec"]
    sleep_enabled = (not args.headless) and (not args.fast) and float(args.speed) > 0.0
    for frame_index, row in enumerate(rows):
        for name in joint_names:
            joint_positions[dof_index[name]] = row["joints"][name]
        articulation.set_joint_positions(joint_positions)
        articulation.set_joint_velocities(zero_velocities)
        articulation.apply_action(ArticulationAction(joint_positions=joint_positions))
        if np.all(np.isfinite(row["target_hand"])):
            target_marker_translate.Set(Gf.Vec3d(*[float(v) for v in row["target_hand"]]))
        should_render = bool(args.record_video) or (not args.headless and frame_index % render_every == 0)
        world.step(render=should_render)
        if args.record_video:
            capture_frame()
        if sleep_enabled and frame_index > 0:
            dt = max(0.0, (row["t_sec"] - previous_t) / float(args.speed))
            if dt > 0.0:
                time.sleep(min(dt, 0.1))
        previous_t = row["t_sec"]
        if frame_index == 0 or (frame_index + 1) % 200 == 0 or frame_index + 1 == len(rows):
            print(
                "[planned-replay] frame %d/%d t=%.3f pose=%s"
                % (frame_index + 1, len(rows), row["t_sec"], row["hand_pose_name"])
            )

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_usd))
    print("[planned-replay] wrote replay scene: %s" % out_usd)

    if args.record_video:
        if frame_index_for_video > 0:
            ffmpeg_bin = shutil.which("ffmpeg")
            if ffmpeg_bin:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                frame_pattern = str(frames_dir / "frame_%06d.png")
                ffmpeg_cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-framerate",
                    "%.6f" % float(args.video_fps),
                    "-i",
                    frame_pattern,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(video_path),
                ]
                proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    print("[planned-replay] video saved: %s" % video_path)
                    if not args.keep_frames:
                        shutil.rmtree(frames_dir, ignore_errors=True)
                        print("[planned-replay] raw frames removed: %s" % frames_dir)
                else:
                    print("[planned-replay] WARN: ffmpeg failed; raw frames kept in %s" % frames_dir)
                    if proc.stderr:
                        err_lines = proc.stderr.strip().splitlines()
                        if err_lines:
                            print("[planned-replay] ffmpeg stderr tail: %s" % err_lines[-1])
            else:
                print("[planned-replay] WARN: ffmpeg not found; raw frames kept in %s" % frames_dir)
        else:
            print("[planned-replay] WARN: no video frames captured")

    if not args.headless and not args.no_preview:
        print("[planned-replay] replay complete. Close Isaac Sim or press Ctrl-C to exit.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
