#!/usr/bin/env python3
"""Set up an Isaac Sim inspection scene for the AUBO i5 + MechHand asset.

This script does not replay motion. It imports the combined robot USD, places it
in the corrected-frame scene, adds visual reference geometry, optionally draws a
corrected trajectory, validates articulation DOFs, and saves a preview USD.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
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


def load_corrected_points(path: Optional[Path], max_points: int = 1500) -> np.ndarray:
    if path is None or not path.is_file():
        return np.zeros((0, 3), dtype=np.float64)
    points: List[List[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                points.append([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
            except (KeyError, TypeError, ValueError):
                continue
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if max_points > 0 and len(arr) > max_points:
        idx = np.linspace(0, len(arr) - 1, max_points).astype(np.int64)
        arr = arr[idx]
    return arr


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--robot-config",
        default=str(_REPO_ROOT / "simulation" / "configs" / "aubo_i5_mechhand.yaml"),
        help="robot asset and placement YAML",
    )
    parser.add_argument(
        "--corrected-trajectory",
        default=None,
        help="optional corrected_trajectory.csv to draw as a magenta reference curve",
    )
    parser.add_argument(
        "--out-usd",
        default=str(_REPO_ROOT / "simulation" / "scenes" / "aubo_i5_mechhand_preview.usd"),
        help="preview scene USD output path",
    )
    parser.add_argument("--prim-path", default=None, help="override robot prim path")
    parser.add_argument("--headless", action="store_true", help="run without viewport")
    parser.add_argument("--no-preview", action="store_true", help="exit after writing/validating the preview scene")
    parser.add_argument("--max-trajectory-points", type=int, default=1500)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config_path = Path(args.robot_config).expanduser().resolve()
    config = load_robot_config(config_path)
    robot_usd = resolve_repo_path(config.get("combined_usd", ""))
    if not robot_usd.is_file():
        raise SystemExit("robot USD not found: %s" % robot_usd)

    out_usd = Path(args.out_usd).expanduser().resolve()
    corrected_points = load_corrected_points(
        Path(args.corrected_trajectory).expanduser().resolve() if args.corrected_trajectory else None,
        max_points=max(0, int(args.max_trajectory_points)),
    )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
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
    print("[scene-preview] opened robot USD: %s" % robot_usd)
    print("[scene-preview] robot prim: %s" % robot_prim_path)

    def make_xform(path: str, xyz: Sequence[float] = (0.0, 0.0, 0.0)):
        xform = UsdGeom.Xform.Define(stage, path)
        UsdGeom.Xformable(xform).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in xyz]))
        return xform

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

    def make_workspace_box(root_path: str, mins: np.ndarray, maxs: np.ndarray):
        corners = np.array(
            [
                [mins[0], mins[1], mins[2]], [maxs[0], mins[1], mins[2]], [maxs[0], maxs[1], mins[2]], [mins[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]], [maxs[0], mins[1], maxs[2]], [maxs[0], maxs[1], maxs[2]], [mins[0], maxs[1], maxs[2]],
            ],
            dtype=np.float64,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        UsdGeom.Xform.Define(stage, root_path)
        for index, (a, b) in enumerate(edges):
            make_polyline("%s/edge_%02d" % (root_path, index), [corners[a], corners[b]], (0.15, 0.75, 1.0), width=0.003)

    def find_articulation_root(root_path: str) -> str:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError("robot prim does not exist: %s" % root_path)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
        return root_path

    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)

    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(2800.0)
    key.CreateAngleAttr(1.0)
    fill = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(450.0)

    world_from_base = config.get("world_from_base", {}) or {}
    base_xyz = np.asarray(world_from_base.get("xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
    base_quat = np.asarray(world_from_base.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)

    make_axis_triad("/World/CorrectedFrame", length=0.28, thick=0.008)
    make_xform("/World/RobotBaseFrame", xyz=base_xyz)
    make_axis_triad("/World/RobotBaseFrame/Triad", length=0.22, thick=0.007)
    make_xform("/World/Table", xyz=(0.0, 0.0, -1.015))
    make_box("/World/Table/surface", (1.2, 0.7, 0.03), (0.50, 0.45, 0.38), opacity=1.0)
    make_workspace_box("/World/WorkspaceBox", np.array([-0.25, -0.20, -0.95]), np.array([0.35, 0.35, -0.55]))

    if len(corrected_points) >= 2:
        make_box("/World/TrajectoryStart", (0.018, 0.018, 0.018), (0.1, 1.0, 0.2), opacity=1.0)
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/TrajectoryStart")).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in corrected_points[0]]))
        make_box("/World/TrajectoryEnd", (0.018, 0.018, 0.018), (1.0, 0.15, 0.1), opacity=1.0)
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/TrajectoryEnd")).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in corrected_points[-1]]))
        print("[scene-preview] trajectory points: %d" % len(corrected_points))

    world = World(stage_units_in_meters=1.0)
    world.reset()
    omni.timeline.get_timeline_interface().play()
    for _ in range(8):
        simulation_app.update()

    articulation_root_path = find_articulation_root(robot_prim_path)
    print("[scene-preview] articulation root: %s" % articulation_root_path)
    articulation = SingleArticulation(prim_path=articulation_root_path)
    articulation.initialize()
    articulation.set_world_pose(position=base_xyz, orientation=base_quat)
    dof_names = list(articulation.dof_names)
    expected_names = list(config.get("arm_joints", [])) + list(config.get("hand_joints", []))
    missing = [name for name in expected_names if name not in dof_names]
    print("[scene-preview] DOF count: %d" % len(dof_names))
    print("[scene-preview] DOFs: %s" % ", ".join(dof_names))
    if missing:
        print("[scene-preview] WARN: missing expected DOFs: %s" % ", ".join(missing))
    else:
        print("[scene-preview] expected DOFs are present")

    joint_targets = np.zeros(len(dof_names), dtype=np.float64)
    default_arm = dict(zip(config.get("arm_joints", []), config.get("default_arm_q", [])))
    default_hand_pose = str(config.get("default_hand_pose", "five_open"))
    try:
        from prism.devices.hand.joint_config import GESTURE_JOINT_CONFIGS
        hand_pose = GESTURE_JOINT_CONFIGS.get(default_hand_pose, {})
    except Exception:
        hand_pose = {}
    prefix = str(config.get("hand_joint_prefix", "mechhand_"))
    for index, name in enumerate(dof_names):
        if name in default_arm:
            joint_targets[index] = float(default_arm[name])
        elif name.startswith(prefix):
            joint_targets[index] = float(hand_pose.get(name[len(prefix):], 0.0))
    articulation.set_joint_positions(joint_targets)
    for _ in range(4):
        simulation_app.update()

    if not args.headless:
        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            target = np.mean(corrected_points, axis=0) if len(corrected_points) else np.array([0.05, 0.10, -0.75], dtype=np.float64)
            cam_state = ViewportCameraState("/OmniverseKit_Persp")
            cam_state.set_position_world(Gf.Vec3d(float(target[0]) + 0.8, float(target[1]) - 1.5, float(target[2]) + 0.8), True)
            cam_state.set_target_world(Gf.Vec3d(float(target[0]), float(target[1]), float(target[2])), True)
        except Exception as exc:
            print("[scene-preview] camera setup skipped: %s" % exc)

    out_usd.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out_usd))
    print("[scene-preview] wrote preview scene: %s" % out_usd)

    if not args.headless and not args.no_preview:
        print("[scene-preview] preview running. Close Isaac Sim or press Ctrl-C to exit.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
