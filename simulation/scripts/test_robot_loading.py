#!/usr/bin/env python3
"""Diagnostic script to test robot USD loading in Isaac Sim."""

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.usd
from isaacsim.core.api import World
from pxr import Usd

# Load config
config_file = _REPO_ROOT / "simulation/configs/aubo_i5_mechhand.yaml"
import yaml
with open(config_file) as f:
    config = yaml.safe_load(f)

robot_usd = str(_REPO_ROOT / "simulation/assets/aubo_i5_mechhand/aubo_i5_mechhand.usd")
robot_prim_path = config.get("robot_prim", "/World/AUBO_i5_MechHand")

print("[TEST] Robot USD:", robot_usd)
print("[TEST] Robot prim path:", robot_prim_path)
print("[TEST] Config keys:", list(config.keys()))

# Create stage
usd_context = omni.usd.get_context()
usd_context.new_stage()
stage = usd_context.get_stage()

# Create Xform root for robot reference
from pxr import UsdGeom, Gf
root = UsdGeom.Xform.Define(stage, robot_prim_path)

# Add reference
root.GetPrim().GetReferences().AddReference(robot_usd)
print("[TEST] Robot reference added to stage")

# List all prims in stage
print("\n[TEST] Prims in stage before physics init:")
for prim in stage.Traverse():
    if prim.IsActive():
        print("  ", prim.GetPath())

# Initialize physics
world = World(stage_units_in_meters=1.0)
world.reset()

print("\n[TEST] Physics initialized")

# Check if robot prim exists
robot_prim = stage.GetPrimAtPath(robot_prim_path)
print("[TEST] Robot prim exists:", robot_prim and robot_prim.IsValid())
print("[TEST] Robot prim:", robot_prim)

# List prims after physics
print("\n[TEST] Prims in stage after physics init (first 20):")
count = 0
for prim in stage.Traverse():
    if prim.IsActive() and "AUBO" in str(prim.GetPath()):
        print("  ", prim.GetPath())
        count += 1
    if count > 20:
        break

# Try to find articulation root
from isaacsim.core.prims import SingleArticulation
try:
    articulation = SingleArticulation(prim_path=robot_prim_path)
    print("\n[TEST] SingleArticulation SUCCESS!")
except Exception as e:
    print("\n[TEST] SingleArticulation FAILED:", e)
    print("[TEST] Attempting to find actual articulation in tree...")
    
    # Search for ArticulationRootAPI
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "AUBO" in path_str and prim.HasAPI(Usd.typed):
            print("  Found typed prim:", path_str)

simulation_app.close()
print("\n[TEST] Done")
