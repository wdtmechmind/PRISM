#!/usr/bin/env python3
"""Import the combined AUBO i5 + MechHand URDF into Isaac Sim as a USD asset.

Run with Isaac Sim's Python:

    /isaac-sim/python.sh simulation/scripts/build_robot_usd.py

By default this opens an Isaac Sim window, imports the combined URDF, saves a USD
asset, prints the articulation DOFs, and keeps the window alive for inspection.
Use ``--headless --no-preview`` for CI-style validation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def patch_package_mesh_uris(urdf_path: Path, patched_path: Path) -> Path:
    """Write a URDF copy whose package:// mesh filenames are absolute paths."""
    src = urdf_path.read_text(encoding="utf-8")

    def replace(match: re.Match) -> str:
        uri = match.group(1)
        if uri.startswith("package://"):
            rel = uri[len("package://"):]
            candidate = _REPO_ROOT / rel
            if candidate.is_file():
                return 'filename="%s"' % str(candidate.resolve())
        if uri.startswith("package:///meshes/"):
            rel = uri[len("package:///meshes/"):]
            candidate = urdf_path.parent / "meshes" / rel
            if candidate.is_file():
                return 'filename="%s"' % str(candidate.resolve())
        return match.group(0)

    patched = re.sub(r'filename="([^"]+)"', replace, src)
    patched_path.parent.mkdir(parents=True, exist_ok=True)
    patched_path.write_text(patched, encoding="utf-8")
    return patched_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--combined-urdf",
        default=str(_REPO_ROOT / "simulation" / "assets" / "aubo_i5_mechhand" / "aubo_i5_mechhand.urdf"),
        help="combined AUBO i5 + MechHand URDF",
    )
    parser.add_argument(
        "--usd-output",
        default=str(_REPO_ROOT / "simulation" / "assets" / "aubo_i5_mechhand" / "aubo_i5_mechhand.usd"),
        help="output USD path",
    )
    parser.add_argument(
        "--prim-path",
        default="/World/AUBO_i5_MechHand",
        help="stage prim path for the imported robot",
    )
    parser.add_argument("--headless", action="store_true", help="run Isaac Sim without a viewport")
    parser.add_argument("--no-preview", action="store_true", help="exit after import/save/validation instead of keeping the window open")
    parser.add_argument("--merge-fixed-joints", action="store_true", help="ask the URDF importer to merge fixed joints")
    parser.add_argument("--no-materials", action="store_true", help="skip robot material/color overrides")
    parser.add_argument("--drive-strength", type=float, default=1.0e6)
    parser.add_argument("--drive-damping", type=float, default=1.0e4)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    combined_urdf = Path(args.combined_urdf).expanduser().resolve()
    usd_output = Path(args.usd_output).expanduser().resolve()
    if not combined_urdf.is_file():
        raise SystemExit("combined URDF not found: %s" % combined_urdf)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    import omni.kit.commands
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    patched_urdf = combined_urdf.with_suffix(".isaac_patched.urdf")
    patch_package_mesh_uris(combined_urdf, patched_urdf)
    print("[robot-usd] patched URDF: %s" % patched_urdf)

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = bool(args.merge_fixed_joints)
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.import_inertia_tensor = True
    import_config.create_physics_scene = False
    import_config.set_default_drive_type(1)
    import_config.set_default_drive_strength(float(args.drive_strength))
    import_config.set_default_position_drive_damping(float(args.drive_damping))

    status, imported_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(patched_urdf),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status or not imported_prim_path:
        raise RuntimeError("URDFParseAndImportFile failed for: %s" % patched_urdf)
    print("[robot-usd] imported articulation root: %s" % imported_prim_path)

    if str(imported_prim_path) != args.prim_path:
        try:
            omni.kit.commands.execute("MovePrim", path_from=str(imported_prim_path), path_to=args.prim_path)
            imported_prim_path = args.prim_path
            print("[robot-usd] moved robot to: %s" % imported_prim_path)
        except Exception as exc:
            print("[robot-usd] WARN: could not move imported prim to %s: %s" % (args.prim_path, exc))

    if not args.no_materials:
        from simulation.common.usd_materials import apply_robot_materials

        counts = apply_robot_materials(stage, str(imported_prim_path))
        print("[robot-usd] material prims: %s" % ", ".join("%s=%d" % item for item in sorted(counts.items())))

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(2500.0)
    light.CreateAngleAttr(1.2)

    dome = UsdLux.DomeLight.Define(stage, "/World/FillLight")
    dome.CreateIntensityAttr(350.0)

    World(stage_units_in_meters=1.0).reset()
    omni.timeline.get_timeline_interface().play()
    for _ in range(4):
        simulation_app.update()

    articulation = SingleArticulation(prim_path=str(imported_prim_path))
    articulation.initialize()
    dof_names = list(articulation.dof_names)
    print("[robot-usd] DOF count: %d" % len(dof_names))
    print("[robot-usd] DOFs: %s" % ", ".join(dof_names))
    expected_arm = ["shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"]
    expected_hand = [
        "mechhand_F1_J1", "mechhand_F1_J2", "mechhand_F1_J3",
        "mechhand_F2_J1", "mechhand_F2_J2",
        "mechhand_F3_J1", "mechhand_F3_J2",
        "mechhand_F4_J1", "mechhand_F4_J2",
        "mechhand_F5_J1", "mechhand_F5_J2",
    ]
    missing = [name for name in expected_arm + expected_hand if name not in dof_names]
    if missing:
        print("[robot-usd] WARN: missing expected DOFs: %s" % ", ".join(missing))
    else:
        print("[robot-usd] expected 17 DOFs are present")

    try:
        articulation.set_joint_positions(np.zeros(len(dof_names), dtype=np.float64))
    except Exception as exc:
        print("[robot-usd] WARN: could not set zero joint positions: %s" % exc)

    usd_output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(usd_output))
    print("[robot-usd] wrote USD: %s" % usd_output)

    if not args.headless and not args.no_preview:
        print("[robot-usd] preview running. Close the Isaac Sim window or press Ctrl-C to exit.")
        try:
            while simulation_app.is_running():
                simulation_app.update()
        except KeyboardInterrupt:
            pass

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
