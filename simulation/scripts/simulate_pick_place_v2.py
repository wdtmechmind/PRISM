#!/usr/bin/env python3
"""Clean pick-place simulation replay with proper object initialization.

Rebuilds the replay logic from scratch with clear separation of:
1. Stage setup (environment, physics)
2. Robot loading
3. Object placement (with explicit xform hierarchy)
4. Motion replay
5. Video recording
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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
    import yaml
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_planned_motion(path: Path, joint_names: Sequence[str]) -> List[dict]:
    """Load planned motion CSV with required columns."""
    rows: List[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = set(reader.fieldnames or [])
        missing = [name for name in joint_names if name not in cols]
        if missing:
            raise SystemExit("planned motion missing joint columns: %s" % ", ".join(missing))
        for row in reader:
            item = {
                "t_sec": float(row.get("t_sec", "0") or 0.0),
                "joints": {name: float(row[name]) for name in joint_names},
                "target_hand": np.array(
                    [
                        float(row.get("target_hand_x_m", "nan")),
                        float(row.get("target_hand_y_m", "nan")),
                        float(row.get("target_hand_z_m", "nan")),
                    ],
                    dtype=np.float64,
                ),
                "hand_pose_name": row.get("hand_pose_name", ""),
            }
            rows.append(item)
    if not rows:
        raise SystemExit("no planned motion rows found: %s" % path)
    return rows


def setup_scene(stage, headless: bool = False):
    """Setup clean scene with physics, lights, and reference geometry."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
    
    # Physics scene
    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    
    # Lighting
    key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key_light.CreateIntensityAttr(2800.0)
    key_light.CreateAngleAttr(1.0)
    
    dome_light = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    dome_light.CreateIntensityAttr(450.0)
    
    print("[simulate-v2] scene: physics and lights configured")


def create_table(stage, table_origin_z: float = -1.015, table_size: Tuple[float, float, float] = (1.2, 0.7, 0.03)):
    """Create a simple table geometry."""
    from pxr import Gf, UsdGeom
    
    # Table transform root
    table_xf = UsdGeom.Xform.Define(stage, "/World/Table")
    table_xformable = UsdGeom.Xformable(table_xf)
    table_xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, table_origin_z))
    
    # Table surface (cube scaled to table dimensions)
    surface = UsdGeom.Cube.Define(stage, "/World/Table/Surface")
    surface.CreateSizeAttr(1.0)
    surface.CreateDisplayColorAttr([Gf.Vec3f(0.50, 0.45, 0.38)])
    surface_xf = UsdGeom.Xformable(surface)
    surface_xf.AddScaleOp().Set(Gf.Vec3f(table_size[0], table_size[1], table_size[2]))
    
    print("[simulate-v2] table: created at z=%.3f" % table_origin_z)
    return table_origin_z


def create_task_object(stage, obj_pos: np.ndarray, obj_size: float = 0.045, 
                      obj_color: Sequence[float] = (0.95, 0.35, 0.10)) -> Tuple:
    """Create a task object (cube) with explicit xform hierarchy.
    
    Returns:
        (xformable_op, prim_path) - the translate op and prim path for updates
    """
    from pxr import Gf, UsdGeom
    
    # Create object xform root at the object position
    obj_root = UsdGeom.Xform.Define(stage, "/World/TaskObject")
    obj_xf = UsdGeom.Xformable(obj_root)
    translate_op = obj_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])))
    
    # Create cube geometry under the xform
    cube = UsdGeom.Cube.Define(stage, "/World/TaskObject/Cube")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in obj_color])])
    cube.CreateDisplayOpacityAttr([1.0])
    
    # Scale cube to desired size
    cube_xf = UsdGeom.Xformable(cube)
    cube_xf.AddScaleOp().Set(Gf.Vec3f(obj_size, obj_size, obj_size))
    
    print("[simulate-v2] object: created at (%.3f, %.3f, %.3f) size=%.3f" 
          % (obj_pos[0], obj_pos[1], obj_pos[2], obj_size))
    
    return translate_op, "/World/TaskObject"


def load_robot(stage, robot_usd: Path, robot_prim: str = "/World/AUBO_i5_MechHand") -> str:
    """Load robot USD and return the articulation root path."""
    from pxr import Sdf
    
    # Reference the robot USD at the specified prim path
    root_prim = stage.DefinePrim(robot_prim, "Xform")
    root_prim.GetReferences().AddReference(str(robot_usd))
    
    print("[simulate-v2] robot: loaded from %s at %s" % (robot_usd.name, robot_prim))
    return robot_prim


def find_articulation_root(stage, root_path: str) -> str:
    """Find the articulation root under the given prim path."""
    from pxr import Usd, UsdPhysics
    
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError("robot prim does not exist: %s" % root_path)
    
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    
    return root_path


def create_overhead_camera(stage, pos: Sequence[float], look_at: Sequence[float]) -> str:
    """Create an overhead camera for recording."""
    from pxr import Gf, UsdGeom
    import math
    
    cam = UsdGeom.Camera.Define(stage, "/World/OverheadCamera")
    xform = UsdGeom.XformCommonAPI(cam)
    
    cam_pos = np.asarray(pos, dtype=np.float64)
    look_at_pt = np.asarray(look_at, dtype=np.float64)
    look_dir = look_at_pt - cam_pos
    look_dir /= max(np.linalg.norm(look_dir), 1e-12)
    
    # Tilt around X axis
    tilt_x_deg = math.degrees(math.asin(float(look_dir[1])))
    
    xform.SetTranslate(Gf.Vec3d(*cam_pos))
    xform.SetRotate(Gf.Vec3f(float(tilt_x_deg), 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    
    print("[simulate-v2] camera: pos=(%.3f, %.3f, %.3f), look_at=(%.3f, %.3f, %.3f)" 
          % (*cam_pos, *look_at_pt))
    
    return "/World/OverheadCamera"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planned-motion", required=True, help="planned_motion.csv file")
    parser.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"),
                        help="robot config YAML")
    parser.add_argument("--out-usd", default=None, help="output USD file path")
    parser.add_argument("--out-video", default=None, help="output video file path")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.2, 0.0])
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, -0.9])
    parser.add_argument("--task-object-size", type=float, default=0.045)
    parser.add_argument("--task-object-color", type=float, nargs=3, default=[0.95, 0.35, 0.10])
    parser.add_argument("--task-object-spawn-x", type=float, default=0.0)
    parser.add_argument("--task-object-spawn-y", type=float, default=0.0)
    parser.add_argument("--task-object-spawn-z", type=float, default=None,
                        help="task object spawn Z; default is on table surface")
    parser.add_argument("--record-video", action="store_true", help="record video during replay")
    parser.add_argument("--video-fps", type=float, default=60.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    
    # Resolve paths
    config_path = Path(args.robot_config).expanduser().resolve()
    config = load_robot_config(config_path)
    planned_motion_path = Path(args.planned_motion).expanduser().resolve()
    
    if not planned_motion_path.is_file():
        raise SystemExit("planned motion file not found: %s" % planned_motion_path)
    
    joint_names = list(config.get("arm_joints", [])) + list(config.get("hand_joints", []))
    if not joint_names:
        raise SystemExit("robot config missing arm_joints/hand_joints")
    
    robot_usd = resolve_repo_path(config.get("combined_usd", ""))
    if not robot_usd.is_file():
        raise SystemExit("robot USD not found: %s" % robot_usd)
    
    # Load motion data
    rows = load_planned_motion(planned_motion_path, joint_names)
    
    # Output paths
    if args.out_usd is None:
        args.out_usd = str(planned_motion_path.parent / "simulation_v2.usd")
    if args.out_video is None:
        args.out_video = str(planned_motion_path.parent / "simulation_v2.mp4")
    
    out_usd = Path(args.out_usd).expanduser().resolve()
    out_video = Path(args.out_video).expanduser().resolve()
    frames_dir = out_video.parent / "frames_v2"
    
    # Initialize Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": bool(args.headless)})
    
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, Sdf, Usd, UsdGeom
    
    # Open a new stage (clean start)
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    
    # Setup scene
    setup_scene(stage, headless=args.headless)
    
    # Create table
    table_origin_z = -1.015
    create_table(stage, table_origin_z)
    
    # Calculate object spawn position
    table_top_z = table_origin_z + 0.5 * 0.03  # table height / 2
    if args.task_object_spawn_z is None:
        spawn_z = table_top_z
    else:
        spawn_z = args.task_object_spawn_z
    
    obj_init_pos = np.array([args.task_object_spawn_x, args.task_object_spawn_y, spawn_z], dtype=np.float64)
    
    # Create task object BEFORE loading robot
    obj_translate_op, obj_prim = create_task_object(
        stage, obj_init_pos, 
        obj_size=args.task_object_size,
        obj_color=args.task_object_color
    )
    
    # Load robot
    robot_prim_path = str(config.get("robot_prim", "/World/AUBO_i5_MechHand"))
    load_robot(stage, robot_usd, robot_prim_path)
    
    # Create camera
    camera_path = create_overhead_camera(stage, args.camera_pos, args.camera_look_at)
    
    # Export USD before simulation
    out_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_usd))
    print("[simulate-v2] exported pre-sim stage: %s" % out_usd)
    
    # Initialize simulation
    world = World(stage_units_in_meters=1.0)
    world.reset()
    omni.timeline.get_timeline_interface().play()
    for _ in range(8):
        simulation_app.update()
    
    # Setup robot
    articulation_root = find_articulation_root(stage, robot_prim_path)
    print("[simulate-v2] articulation root: %s" % articulation_root)
    
    articulation = SingleArticulation(prim_path=articulation_root)
    articulation.initialize()
    
    world_from_base = config.get("world_from_base", {}) or {}
    base_xyz = np.asarray(world_from_base.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    base_quat = np.asarray(world_from_base.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    articulation.set_world_pose(position=base_xyz, orientation=base_quat)
    
    dof_names = list(articulation.dof_names)
    dof_index = {name: idx for idx, name in enumerate(dof_names)}
    
    missing = [n for n in joint_names if n not in dof_index]
    if missing:
        raise SystemExit("robot missing DOFs: %s" % ", ".join(missing))
    
    print("[simulate-v2] robot: %d DOFs loaded" % len(dof_names))
    print("[simulate-v2] motion: %d frames to replay" % len(rows))
    
    # Video recording setup
    collector_viewport = None
    frame_count = 0
    if args.record_video:
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            from isaacsim.core.utils.viewports import set_active_viewport_camera, get_active_viewport
            set_active_viewport_camera(camera_path)
            collector_viewport = get_active_viewport()
            print("[simulate-v2] video: recording to %s" % frames_dir)
        except Exception as e:
            print("[simulate-v2] WARNING: video setup failed: %s" % e)
    
    def capture_frame():
        nonlocal frame_count
        if collector_viewport is None:
            return
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file
            out_png = frames_dir / ("frame_%06d.png" % frame_count)
            capture_viewport_to_file(collector_viewport, str(out_png))
            frame_count += 1
        except Exception as e:
            print("[simulate-v2] frame capture failed: %s" % e)
    
    # Replay motion
    joint_positions = np.zeros(len(dof_names), dtype=np.float64)
    zero_velocities = np.zeros(len(dof_names), dtype=np.float64)
    
    print("[simulate-v2] replaying motion...")
    prev_t = rows[0]["t_sec"]
    
    for frame_idx, row in enumerate(rows):
        # Set joint positions
        for name in joint_names:
            joint_positions[dof_index[name]] = row["joints"][name]
        
        articulation.set_joint_positions(joint_positions)
        articulation.set_joint_velocities(zero_velocities)
        articulation.apply_action(ArticulationAction(joint_positions=joint_positions))
        
        # Step simulation
        world.step(render=bool(args.record_video) or (not args.headless))
        
        # Capture frame
        if args.record_video:
            capture_frame()
        
        # Progress logging
        if frame_idx % max(1, len(rows) // 4) == 0 or frame_idx == len(rows) - 1:
            print("[simulate-v2] frame %d/%d (t=%.2f s)" % (frame_idx + 1, len(rows), row["t_sec"]))
        
        # Sleep for realtime playback
        if not args.headless and frame_idx > 0:
            dt = (row["t_sec"] - prev_t) / args.speed
            if dt > 0:
                time.sleep(min(dt, 0.1))
        prev_t = row["t_sec"]
    
    print("[simulate-v2] replay complete")
    
    print("\n[simulate-v2] === CONTINUOUS LOOP - Close window to exit ===\n")
    print("[simulate-v2] Keep the window open to observe the final state.\n")
    
    loop_frame = 0
    while simulation_app.is_running():
        world.step(render=True)
        loop_frame += 1
        if loop_frame % 300 == 0:  # Print every ~5 seconds at 60 FPS
            print("[simulate-v2] Looping... frames: %d" % loop_frame)
    
    print("\n[simulate-v2] Isaac Sim closed by user")
    
    # Video export
    if args.record_video and frame_count > 0:
        try:
            import shutil
            import subprocess
            
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                out_video.parent.mkdir(parents=True, exist_ok=True)
                frame_pattern = str(frames_dir / "frame_%06d.png")
                cmd = [
                    ffmpeg, "-y", "-framerate", str(args.video_fps),
                    "-i", frame_pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(out_video)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print("[simulate-v2] video saved: %s" % out_video)
                    shutil.rmtree(frames_dir, ignore_errors=True)
                else:
                    print("[simulate-v2] ffmpeg failed: %s" % result.stderr)
        except Exception as e:
            print("[simulate-v2] video export failed: %s" % e)
    
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
