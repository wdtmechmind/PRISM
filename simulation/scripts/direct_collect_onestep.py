#!/usr/bin/env python3
"""One-step simulation data collection - generate trajectory and collect data directly.

Single command:
  python direct_collect_onestep.py --pick 0 0 -0.9 --place 0.2 0 -0.9 --num-trials 2
"""

from __future__ import annotations

print("[STARTUP] Script started", flush=True)

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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


def create_place_indicator(stage, place_xyz: np.ndarray, radius: float = 0.1):
    """Create a visual indicator (circle) for the place location."""
    from pxr import Gf, UsdGeom
    
    # Create a cylinder as a flat circle indicator
    indicator = UsdGeom.Cylinder.Define(stage, "/World/PlaceIndicator")
    indicator.CreateHeightAttr(0.01)  # Very thin
    indicator.CreateRadiusAttr(radius)
    indicator.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.8, 0.2)])  # Green
    indicator.CreateDisplayOpacityAttr([0.3])  # Semi-transparent
    
    # Position it at place location
    ind_xf = UsdGeom.Xformable(indicator)
    ind_xf.AddTranslateOp().Set(Gf.Vec3d(float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2]) + 0.01))
    
    return indicator


def generate_pick_place_trajectory(
    pick_xyz: np.ndarray,
    place_xyz: np.ndarray,
    fps: float = 30.0,
    approach_sec: float = 0.5,
    grasp_sec: float = 0.3,
    lift_sec: float = 0.8,
    transfer_sec: float = 1.4,
    release_sec: float = 0.5,
) -> List[dict]:
    """Generate a simple pick-place trajectory.
    
    Stages:
    1. Approach to pick (0.5s)
    2. Grasp at pick (0.3s)
    3. Lift from pick (0.8s)
    4. Transfer to place (1.4s)
    5. Release at place (0.5s)
    """
    
    trajectory = []
    frame_count = 0
    t_sec = 0.0
    dt = 1.0 / fps
    
    pick_approach = pick_xyz + np.array([0.0, 0.0, 0.1], dtype=np.float64)
    place_approach = place_xyz + np.array([0.0, 0.0, 0.1], dtype=np.float64)
    
    stages = [
        ("approach_pick", pick_approach, pick_xyz, approach_sec),
        ("grasp_pick", pick_xyz, pick_xyz, grasp_sec),
        ("lift", pick_xyz, place_approach, lift_sec),
        ("transfer", place_approach, place_xyz, transfer_sec),
        ("release", place_xyz, place_xyz, release_sec),
    ]
    
    for stage_name, start_pos, end_pos, duration_sec in stages:
        num_frames = max(1, int(duration_sec * fps))
        for i in range(num_frames):
            alpha = i / num_frames if num_frames > 1 else 1.0
            current_pos = start_pos * (1.0 - alpha) + end_pos * alpha
            
            trajectory.append({
                "frame_index": frame_count,
                "t_sec": t_sec,
                "stage": stage_name,
                "x_m": float(current_pos[0]),
                "y_m": float(current_pos[1]),
                "z_m": float(current_pos[2]),
            })
            
            frame_count += 1
            t_sec += dt
    
    return trajectory


def setup_scene(stage):
    """Setup scene with physics and lights."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
    
    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    
    key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key_light.CreateIntensityAttr(2800.0)
    key_light.CreateAngleAttr(1.0)
    
    dome_light = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    dome_light.CreateIntensityAttr(450.0)
    
    print("[onestep] scene: physics configured", flush=True)


def create_table(stage, table_origin_z: float = -1.015):
    """Create table."""
    from pxr import Gf, UsdGeom
    
    table_xf = UsdGeom.Xform.Define(stage, "/World/Table")
    table_xf_op = UsdGeom.Xformable(table_xf)
    table_xf_op.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, table_origin_z))
    
    surface = UsdGeom.Cube.Define(stage, "/World/Table/Surface")
    surface.CreateSizeAttr(1.0)
    surface.CreateDisplayColorAttr([Gf.Vec3f(0.50, 0.45, 0.38)])
    surf_xf = UsdGeom.Xformable(surface)
    surf_xf.AddScaleOp().Set(Gf.Vec3f(1.2, 0.7, 0.03))
    
    return table_origin_z


def create_task_object(stage, obj_pos: np.ndarray, obj_size: float = 0.045):
    """Create task object."""
    from pxr import Gf, UsdGeom
    
    obj_root = UsdGeom.Xform.Define(stage, "/World/TaskObject")
    obj_xf = UsdGeom.Xformable(obj_root)
    translate_op = obj_xf.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])))
    
    cube = UsdGeom.Cube.Define(stage, "/World/TaskObject/Cube")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.10)])
    cube_xf = UsdGeom.Xformable(cube)
    cube_xf.AddScaleOp().Set(Gf.Vec3f(obj_size, obj_size, obj_size))
    
    return translate_op, obj_pos.copy()


def load_robot(stage, robot_usd: Path, robot_prim: str = "/World/AUBO_i5_MechHand"):
    """Load robot."""
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Task parameters
    parser.add_argument("--pick", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                        help="pick location (meters)")
    parser.add_argument("--place", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                        help="place location (meters)")
    
    parser.add_argument("--num-trials", type=int, default=1, help="number of trials to collect")
    parser.add_argument("--fps", type=float, default=30.0, help="trajectory FPS")
    
    # Timing
    parser.add_argument("--approach-sec", type=float, default=0.5)
    parser.add_argument("--grasp-sec", type=float, default=0.3)
    parser.add_argument("--lift-sec", type=float, default=0.8)
    parser.add_argument("--transfer-sec", type=float, default=1.4)
    parser.add_argument("--release-sec", type=float, default=0.5)
    
    # Object
    parser.add_argument("--obj-size", type=float, default=0.045)
    parser.add_argument("--obj-x", type=float, default=0.0, help="initial object X")
    parser.add_argument("--obj-y", type=float, default=0.0, help="initial object Y")
    parser.add_argument("--obj-z", type=float, default=None, help="initial object Z (default: table top)")
    
    # Robot config
    parser.add_argument("--robot-config", 
                        default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"))
    
    # Output
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "data" / "sim_collection"),
                        help="output directory for collected data")
    parser.add_argument("--task-name", default="pick_place", help="task name for output")
    
    # Display
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-loop", action="store_true", help="exit immediately after collection (don't loop)")
    
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    print("[DEBUG] main() called", flush=True)
    args = build_arg_parser().parse_args(argv)
    print("[DEBUG] args parsed", flush=True)
    
    # Setup paths
    print("[DEBUG] loading robot config", flush=True)
    config_path = Path(args.robot_config).expanduser().resolve()
    print("[DEBUG] config_path resolved", flush=True)
    config = load_robot_config(config_path)
    print("[DEBUG] config loaded", flush=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[DEBUG] output_dir created", flush=True)
    
    joint_names = list(config.get("arm_joints", [])) + list(config.get("hand_joints", []))
    robot_usd = resolve_repo_path(config.get("combined_usd", ""))
    print("[DEBUG] robot_usd resolved", flush=True)
    
    pick_pos = np.array(args.pick, dtype=np.float64)
    place_pos = np.array(args.place, dtype=np.float64)
    print("[DEBUG] pick/place arrays created", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("[onestep] ONE-STEP SIMULATION DATA COLLECTION", flush=True)
    print("="*60, flush=True)
    print("[onestep] Task: %s" % args.task_name, flush=True)
    print("[onestep] Pick:  (%.3f, %.3f, %.3f)" % tuple(pick_pos), flush=True)
    print("[onestep] Place: (%.3f, %.3f, %.3f)" % tuple(place_pos), flush=True)
    print("[onestep] Trials: %d (randomized=%s)" % (args.num_trials, args.num_trials > 1), flush=True)
    print("[onestep] Output: %s" % output_dir, flush=True)
    print("="*60 + "\n", flush=True)
    
    # Generate reference trajectory (for metadata and timing info)
    print("[onestep] generating reference trajectory...", flush=True)
    trajectory = generate_pick_place_trajectory(
        pick_pos, place_pos,
        fps=args.fps,
        approach_sec=args.approach_sec,
        grasp_sec=args.grasp_sec,
        lift_sec=args.lift_sec,
        transfer_sec=args.transfer_sec,
        release_sec=args.release_sec,
    )
    print("[onestep] trajectory: %d frames (%.1f seconds)" % (len(trajectory), trajectory[-1]["t_sec"]))
    
    # Initialize Isaac Sim
    from isaacsim import SimulationApp
    # Monkey-patch _wait_for_viewport to avoid infinite hang when no GPU renderer
    def _patched_wait_for_viewport(self) -> None:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport_api = get_active_viewport()
            max_frames = 30  # Max 30 update cycles to wait for viewport
            frame = 0
            while frame < max_frames and viewport_api.frame_info.get("viewport_handle", None) is None:
                self._app.update()
                frame += 1
        except Exception:
            pass
        for _ in range(10):
            self._app.update()
    SimulationApp._wait_for_viewport = _patched_wait_for_viewport
    
    simulation_app = SimulationApp({"headless": False})
    
    # After SimulationApp(), Isaac Sim redirects stdout. Use file log for debug.
    import sys as _sys
    _debug_log = open(str(output_dir / "debug_run.log"), "w", buffering=1)
    def _log(msg):
        _sys.__stdout__.write(msg + "\n")
        _sys.__stdout__.flush()
        _debug_log.write(msg + "\n")
        _debug_log.flush()
    
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import Usd, UsdGeom
    
    _log("[onestep] Isaac Sim initialized")
    
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    
    # Setup scene
    setup_scene(stage)
    table_origin_z = create_table(stage)
    
    # Object position
    table_top_z = table_origin_z + 0.5 * 0.03
    obj_z = args.obj_z if args.obj_z is not None else table_top_z
    obj_init_pos = np.array([args.obj_x, args.obj_y, obj_z], dtype=np.float64)
    
    obj_translate_op, obj_pos = create_task_object(stage, obj_init_pos, args.obj_size)
    _log("[onestep] object: pos=(%.3f, %.3f, %.3f) size=%.3f" 
          % (obj_init_pos[0], obj_init_pos[1], obj_init_pos[2], args.obj_size))
    
    # Create initial place indicator
    create_place_indicator(stage, place_pos, radius=0.1)
    _log("[onestep] place indicator: created at (%.3f, %.3f, %.3f)" % tuple(place_pos))
    
    # Load robot
    robot_prim_path = str(config.get("robot_prim", "/World/AUBO_i5_MechHand"))
    load_robot(stage, robot_usd, robot_prim_path)
    
    # Initialize simulation
    world = World(stage_units_in_meters=1.0)
    world.reset()
    omni.timeline.get_timeline_interface().play()
    
    _log("[onestep] simulation initialized, stepping 20 frames for robot load...")
    # Give physics system time to load the robot USD
    for _ in range(20):
        simulation_app.update()
    
    # Setup robot (visual reference only - no articulation control)
    _log("[onestep] robot loaded for visual reference (no joint control)")
    
    # Collect trials
    all_trials_data = []
    rng = np.random.default_rng(args.num_trials)  # Random number generator for pick/place variation
    
    for trial_id in range(1, args.num_trials + 1):
        # Randomize pick and place locations for each trial
        if args.num_trials > 1:
            # Random variation around base positions
            pick_variation = np.array([
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                0.0
            ], dtype=np.float64)
            place_variation = np.array([
                rng.uniform(-0.05, 0.05),
                rng.uniform(-0.05, 0.05),
                0.0
            ], dtype=np.float64)
            
            trial_pick = pick_pos + pick_variation
            trial_place = place_pos + place_variation
        else:
            trial_pick = pick_pos
            trial_place = place_pos
        
        _log("\n[onestep] ========== TRIAL %d/%d ==========" % (trial_id, args.num_trials))
        _log("[onestep] pick:  (%.3f, %.3f, %.3f)" % tuple(trial_pick))
        _log("[onestep] place: (%.3f, %.3f, %.3f)" % tuple(trial_place))
        
        # Update place indicator
        place_indicator_path = "/World/PlaceIndicator"
        place_indicator_prim = stage.GetPrimAtPath(place_indicator_path)
        if place_indicator_prim and place_indicator_prim.IsValid():
            from pxr import Sdf
            stage.RemovePrim(Sdf.Path(place_indicator_path))
        
        create_place_indicator(stage, trial_place, radius=0.1)
        
        # Generate trajectory for this trial
        trial_trajectory = generate_pick_place_trajectory(
            trial_pick, trial_place,
            fps=args.fps,
            approach_sec=args.approach_sec,
            grasp_sec=args.grasp_sec,
            lift_sec=args.lift_sec,
            transfer_sec=args.transfer_sec,
            release_sec=args.release_sec,
        )
        
        trial_data = []
        
        wall_time_start = time.time()
        
        for frame_idx, traj_point in enumerate(trial_trajectory):
            # Step simulation (object falls under gravity)
            world.step(render=not args.headless)
            
            # Record data
            wall_time_now = time.time()
            
            data_row = {
                "trial_id": trial_id,
                "frame_index": frame_idx,
                "t_sec": float(traj_point["t_sec"]),
                "stage": traj_point["stage"],
                "hand_target_x_m": float(traj_point["x_m"]),
                "hand_target_y_m": float(traj_point["y_m"]),
                "hand_target_z_m": float(traj_point["z_m"]),
                "pick_x_m": float(trial_pick[0]),
                "pick_y_m": float(trial_pick[1]),
                "pick_z_m": float(trial_pick[2]),
                "place_x_m": float(trial_place[0]),
                "place_y_m": float(trial_place[1]),
                "place_z_m": float(trial_place[2]),
                "object_x_m": float(obj_pos[0]),
                "object_y_m": float(obj_pos[1]),
                "object_z_m": float(obj_pos[2]),
                "wall_time": float(wall_time_now - wall_time_start),
            }
            
            trial_data.append(data_row)
            
            if (frame_idx + 1) % max(1, len(trial_trajectory) // 4) == 0 or frame_idx == len(trial_trajectory) - 1:
                _log("[onestep] trial %d: frame %d/%d (wall=%.1fs)" % (trial_id, frame_idx + 1, len(trial_trajectory), time.time() - wall_time_start))
        
        all_trials_data.extend(trial_data)
        _log("[onestep] trial %d: collected %d frames" % (trial_id, len(trial_data)))
    
    # Save all data
    _log("\n[onestep] saving data...")
    
    output_csv = output_dir / ("%s_all_trials.csv" % args.task_name)
    if all_trials_data:
        fieldnames = list(all_trials_data[0].keys())
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trials_data)
        _log("[onestep] data saved: %s (%d frames)" % (output_csv, len(all_trials_data)))
    
    # Save metadata
    metadata = {
        "task_name": args.task_name,
        "base_pick_xyz": args.pick,
        "base_place_xyz": args.place,
        "num_trials": args.num_trials,
        "trials_randomized": args.num_trials > 1,
        "total_frames": len(all_trials_data),
        "object_initial": [args.obj_x, args.obj_y, obj_z],
        "fps": args.fps,
        "trajectory_duration_sec": trajectory[-1]["t_sec"] if trajectory else 0.0,
        "robot_config": str(config_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    metadata_path = output_dir / ("%s_metadata.json" % args.task_name)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    _log("[onestep] metadata saved: %s" % metadata_path)
    
    _log("\n[onestep] ========== DATA COLLECTION COMPLETE ==========")
    _log("[onestep] Output directory: %s" % output_dir)
    _debug_log.close()
    
    if not args.no_loop:
        print("[onestep] Entering observation loop (close window to exit)...\n", flush=True)
        loop_frame = 0
        while simulation_app.is_running():
            world.step(render=True)
            loop_frame += 1
            if loop_frame % 300 == 0:
                print("[onestep] observation: %d frames" % loop_frame)
    
    simulation_app.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
