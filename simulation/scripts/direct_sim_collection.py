#!/usr/bin/env python3
"""Direct simulation data collection - collect data while simulating motion.

Unlike replay scripts, this collects data during the simulation:
- Robot joint positions and velocities
- Task object position and orientation
- Sensor data (if available)
- One-pass data gathering without post-replay
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


def load_trajectory_csv(path: Path, max_points: int = 0) -> List[dict]:
    """Load corrected_trajectory.csv with hand positions."""
    if not path.is_file():
        raise SystemExit("trajectory CSV not found: %s" % path)
    
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                item = {
                    "t_sec": float(row.get("t_sec", "0") or 0.0),
                    "t_trial": float(row.get("t_trial", "0") or 0.0),
                    "x_m": float(row.get("x_m", "0") or 0.0),
                    "y_m": float(row.get("y_m", "0") or 0.0),
                    "z_m": float(row.get("z_m", "0") or 0.0),
                }
                rows.append(item)
            except (ValueError, TypeError):
                continue
    
    if not rows:
        raise SystemExit("no trajectory points found: %s" % path)
    
    if max_points > 0 and len(rows) > max_points:
        idx = np.linspace(0, len(rows) - 1, max_points).astype(int)
        rows = [rows[i] for i in idx]
    
    return rows


def setup_scene(stage, headless: bool = False):
    """Setup scene with physics, lights."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
    
    # Physics
    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    
    # Lights
    key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key_light.CreateIntensityAttr(2800.0)
    key_light.CreateAngleAttr(1.0)
    
    dome_light = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    dome_light.CreateIntensityAttr(450.0)
    
    print("[direct-collect] scene: physics and lights configured")


def create_table(stage, table_origin_z: float = -1.015, table_size: Tuple[float, float, float] = (1.2, 0.7, 0.03)):
    """Create table geometry."""
    from pxr import Gf, UsdGeom
    
    table_xf = UsdGeom.Xform.Define(stage, "/World/Table")
    table_xformable = UsdGeom.Xformable(table_xf)
    table_xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, table_origin_z))
    
    surface = UsdGeom.Cube.Define(stage, "/World/Table/Surface")
    surface.CreateSizeAttr(1.0)
    surface.CreateDisplayColorAttr([Gf.Vec3f(0.50, 0.45, 0.38)])
    surface_xf = UsdGeom.Xformable(surface)
    surface_xf.AddScaleOp().Set(Gf.Vec3f(table_size[0], table_size[1], table_size[2]))
    
    return table_origin_z


def create_task_object(stage, obj_pos: np.ndarray, obj_size: float = 0.045, 
                      obj_color: Sequence[float] = (0.95, 0.35, 0.10)):
    """Create task object and return xform operator."""
    from pxr import Gf, UsdGeom
    
    obj_root = UsdGeom.Xform.Define(stage, "/World/TaskObject")
    obj_xf = UsdGeom.Xformable(obj_root)
    translate_op = obj_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])))
    
    cube = UsdGeom.Cube.Define(stage, "/World/TaskObject/Cube")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*[float(v) for v in obj_color])])
    cube.CreateDisplayOpacityAttr([1.0])
    
    cube_xf = UsdGeom.Xformable(cube)
    cube_xf.AddScaleOp().Set(Gf.Vec3f(obj_size, obj_size, obj_size))
    
    return translate_op, obj_pos.copy()


def load_robot(stage, robot_usd: Path, robot_prim: str = "/World/AUBO_i5_MechHand") -> str:
    """Load robot USD."""
    root_prim = stage.DefinePrim(robot_prim, "Xform")
    root_prim.GetReferences().AddReference(str(robot_usd))
    return robot_prim


def find_articulation_root(stage, root_path: str) -> str:
    """Find articulation root."""
    from pxr import Usd, UsdPhysics
    
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError("robot prim does not exist: %s" % root_path)
    
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    
    return root_path


def compute_ik_target_from_trajectory(trajectory_point: dict) -> np.ndarray:
    """Extract hand target position from trajectory point."""
    return np.array([
        trajectory_point["x_m"],
        trajectory_point["y_m"],
        trajectory_point["z_m"],
    ], dtype=np.float64)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-trajectory", required=True, help="corrected_trajectory.csv file")
    parser.add_argument("--robot-config", default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    parser.add_argument("--output-dir", required=True, help="output directory for collected data")
    parser.add_argument("--trial-name", default="trial_001", help="trial identifier")
    
    parser.add_argument("--task-object-size", type=float, default=0.045)
    parser.add_argument("--task-object-color", type=float, nargs=3, default=[0.95, 0.35, 0.10])
    parser.add_argument("--task-object-x", type=float, default=0.0, help="initial object X position")
    parser.add_argument("--task-object-y", type=float, default=0.0, help="initial object Y position")
    parser.add_argument("--task-object-z", type=float, default=None, help="initial object Z position (default: table top)")
    
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.2, 0.0])
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, -0.9])
    
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    
    # Setup paths
    config_path = Path(args.robot_config).expanduser().resolve()
    config = load_robot_config(config_path)
    traj_path = Path(args.corrected_trajectory).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    trajectory = load_trajectory_csv(traj_path)
    
    joint_names = list(config.get("arm_joints", [])) + list(config.get("hand_joints", []))
    if not joint_names:
        raise SystemExit("robot config missing arm_joints/hand_joints")
    
    robot_usd = resolve_repo_path(config.get("combined_usd", ""))
    if not robot_usd.is_file():
        raise SystemExit("robot USD not found: %s" % robot_usd)
    
    print("[direct-collect] loaded trajectory with %d points" % len(trajectory))
    print("[direct-collect] trial: %s" % args.trial_name)
    
    # Initialize Isaac Sim
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": bool(args.headless)})
    
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Gf, Usd, UsdGeom
    
    # Create stage
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    
    # Setup scene
    setup_scene(stage, headless=args.headless)
    table_origin_z = create_table(stage)
    
    # Setup task object
    table_top_z = table_origin_z + 0.5 * 0.03
    if args.task_object_z is None:
        obj_spawn_z = table_top_z
    else:
        obj_spawn_z = args.task_object_z
    
    obj_init_pos = np.array([args.task_object_x, args.task_object_y, obj_spawn_z], dtype=np.float64)
    obj_translate_op, obj_pos = create_task_object(
        stage, obj_init_pos,
        obj_size=args.task_object_size,
        obj_color=args.task_object_color
    )
    
    # Load robot
    robot_prim_path = str(config.get("robot_prim", "/World/AUBO_i5_MechHand"))
    load_robot(stage, robot_usd, robot_prim_path)
    
    # Initialize simulation
    world = World(stage_units_in_meters=1.0)
    world.reset()
    omni.timeline.get_timeline_interface().play()
    for _ in range(8):
        simulation_app.update()
    
    # Setup articulation
    articulation_root = find_articulation_root(stage, robot_prim_path)
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
    
    print("[direct-collect] robot: %d DOFs" % len(dof_names))
    
    # Data collection lists
    collected_data = []
    
    # Simplified IK approach: interpolate from hand targets
    # In a real system, you'd use a proper IK solver
    print("[direct-collect] collecting data from %d trajectory points" % len(trajectory))
    
    joint_positions = np.zeros(len(dof_names), dtype=np.float64)
    zero_velocities = np.zeros(len(dof_names), dtype=np.float64)
    
    wall_time_start = time.time()
    
    for frame_idx, traj_point in enumerate(trajectory):
        # Get hand target
        hand_target = compute_ik_target_from_trajectory(traj_point)
        
        # For now, just hold default posture (in real system, would solve IK)
        # This is a simplified version - you would integrate your IK solver here
        for idx, name in enumerate(dof_names):
            if name in config.get("arm_joints", []):
                # Default arm posture
                default_arm = config.get("default_arm_q", [])
                if idx < len(default_arm):
                    joint_positions[idx] = default_arm[idx]
        
        # Apply joint positions
        articulation.set_joint_positions(joint_positions)
        articulation.set_joint_velocities(zero_velocities)
        articulation.apply_action(ArticulationAction(joint_positions=joint_positions))
        
        # Step simulation
        world.step(render=not args.headless)
        
        # Record data
        wall_time_now = time.time()
        wall_time_sec = wall_time_now - wall_time_start
        
        # Get current joint positions
        current_positions = articulation.get_joint_positions().tolist()
        current_velocities = articulation.get_joint_velocities().tolist()
        
        # Get end effector position (approx from hand target)
        ee_pos = hand_target
        
        data_row = {
            "frame_index": frame_idx,
            "t_sec": float(traj_point["t_sec"]),
            "t_trial": float(traj_point["t_trial"]),
            "wall_time": wall_time_sec,
            "hand_target_x_m": float(hand_target[0]),
            "hand_target_y_m": float(hand_target[1]),
            "hand_target_z_m": float(hand_target[2]),
            "object_x_m": float(obj_pos[0]),
            "object_y_m": float(obj_pos[1]),
            "object_z_m": float(obj_pos[2]),
        }
        
        # Add joint positions and velocities
        for joint_name, joint_idx in dof_index.items():
            data_row[f"{joint_name}_pos_rad"] = current_positions[joint_idx]
            data_row[f"{joint_name}_vel_rad_s"] = current_velocities[joint_idx]
        
        collected_data.append(data_row)
        
        # Progress
        if (frame_idx + 1) % max(1, len(trajectory) // 4) == 0 or frame_idx == len(trajectory) - 1:
            print("[direct-collect] frame %d/%d collected" % (frame_idx + 1, len(trajectory)))
    
    print("[direct-collect] data collection complete: %d frames" % len(collected_data))
    
    # Save collected data
    output_csv = output_dir / ("%s_collected_data.csv" % args.trial_name)
    if collected_data:
        fieldnames = list(collected_data[0].keys())
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(collected_data)
        print("[direct-collect] data saved: %s" % output_csv)
    
    # Save metadata
    metadata_path = output_dir / ("%s_metadata.json" % args.trial_name)
    import json
    metadata = {
        "trial_name": args.trial_name,
        "trajectory_file": str(traj_path),
        "collected_frames": len(collected_data),
        "robot_config": str(config_path),
        "object_initial_position": obj_init_pos.tolist(),
        "object_size": args.task_object_size,
        "table_top_z": table_top_z,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[direct-collect] metadata saved: %s" % metadata_path)
    
    # Loop for observation
    print("\n[direct-collect] === DATA COLLECTION COMPLETE ===")
    print("[direct-collect] Data files saved to: %s" % output_dir)
    print("[direct-collect] Entering observation loop (close window to exit)...\n")
    
    loop_frame = 0
    while simulation_app.is_running():
        world.step(render=True)
        loop_frame += 1
        if loop_frame % 300 == 0:
            print("[direct-collect] observation loop: %d frames" % loop_frame)
    
    print("\n[direct-collect] Isaac Sim closed by user")
    simulation_app.close()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
