#!/usr/bin/env python3
"""Test object placement without complex robot motion."""

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main():
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdLux
    
    # Create new stage
    usd_context = omni.usd.get_context()
    usd_context.new_stage()
    stage = usd_context.get_stage()
    
    print("\n=== VALIDATION: Object Placement ===\n")
    
    # Physics
    physics = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/PhysicsScene"))
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    
    # Lighting
    key = UsdLux.DistantLight.Define(stage, "/World/Light")
    key.CreateIntensityAttr(2000.0)
    
    # TABLE
    print("1. Creating table...")
    table_z = -1.015
    table_xf = UsdGeom.Xform.Define(stage, "/World/Table")
    table_xf_op = UsdGeom.Xformable(table_xf)
    table_xf_op.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, table_z))
    
    table_surf = UsdGeom.Cube.Define(stage, "/World/Table/Surface")
    table_surf.CreateSizeAttr(1.0)
    table_surf.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.45, 0.38)])
    table_surf_xf = UsdGeom.Xformable(table_surf)
    table_surf_xf.AddScaleOp().Set(Gf.Vec3f(1.2, 0.7, 0.03))
    
    table_top_z = table_z + 0.5 * 0.03
    print("   Table top Z = %.4f" % table_top_z)
    
    # OBJECT AT TABLE TOP
    print("2. Creating object at table surface...")
    obj_x, obj_y = 0.0, 0.0
    obj_z = table_top_z
    obj_size = 0.045
    
    obj_xf = UsdGeom.Xform.Define(stage, "/World/TaskObject")
    obj_xf_op = UsdGeom.Xformable(obj_xf)
    obj_translate = obj_xf_op.AddTranslateOp()
    obj_translate.Set(Gf.Vec3d(obj_x, obj_y, obj_z))
    
    obj_cube = UsdGeom.Cube.Define(stage, "/World/TaskObject/Cube")
    obj_cube.CreateSizeAttr(1.0)
    obj_cube.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.35, 0.10)])
    obj_cube_xf = UsdGeom.Xformable(obj_cube)
    obj_cube_xf.AddScaleOp().Set(Gf.Vec3f(obj_size, obj_size, obj_size))
    
    print("   Object position: (%.4f, %.4f, %.4f)" % (obj_x, obj_y, obj_z))
    print("   Object size: %.4f" % obj_size)
    
    # REFERENCE MARKERS
    print("3. Creating reference markers...")
    
    # Origin marker (red)
    origin = UsdGeom.Cube.Define(stage, "/World/OriginMarker")
    origin.CreateSizeAttr(1.0)
    origin.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])
    origin_xf = UsdGeom.Xformable(origin)
    origin_xf.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.05))
    print("   Origin (0, 0, 0) = RED cube at 0,0,0")
    
    # Table top marker (green)
    table_marker = UsdGeom.Cube.Define(stage, "/World/TableTopMarker")
    table_marker.CreateSizeAttr(1.0)
    table_marker.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.0)])
    table_marker_xf = UsdGeom.Xformable(table_marker)
    table_marker_xf.AddTranslateOp().Set(Gf.Vec3d(0.2, 0.0, table_top_z))
    table_marker_xf.AddScaleOp().Set(Gf.Vec3f(0.05, 0.05, 0.05))
    print("   Table top (0.2, 0, %.4f) = GREEN cube" % table_top_z)
    
    print("\n4. Initializing physics world...")
    world = World(stage_units_in_meters=1.0)
    world.reset()
    omni.timeline.get_timeline_interface().play()
    
    for _ in range(8):
        simulation_app.update()
    
    print("   World initialized")
    
    print("\n5. Running simulation (CONTINUOUS LOOP)...")
    print("   Look for:")
    print("   - ORANGE cube: should rest on table (not float above or sink below)")
    print("   - RED cube: origin reference")
    print("   - GREEN cube: table surface reference")
    print("   \n   === KEEP WINDOW OPEN TO OBSERVE ===\n")
    print("   Close the Isaac Sim window to exit this script.\n")
    
    frame_count = 0
    while simulation_app.is_running():
        world.step(render=True)
        frame_count += 1
        if frame_count % 300 == 0:  # Print every ~5 seconds at 60 FPS
            print("   [Running... frames: %d]" % frame_count)
    
    print("\n=== Isaac Sim closed by user ===\n")
    simulation_app.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
