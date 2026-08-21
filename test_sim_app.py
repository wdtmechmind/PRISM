#!/usr/bin/env python3
print("[IMPORT_TEST] Start", flush=True)

print("[IMPORT_TEST] Before isaacsim import", flush=True)
from isaacsim import SimulationApp
print("[IMPORT_TEST] After isaacsim import", flush=True)

print("[IMPORT_TEST] Creating SimulationApp", flush=True)
app = SimulationApp({"headless": True})
print("[IMPORT_TEST] SimulationApp created", flush=True)

import sys
sys.stdout.flush()
sys.stderr.flush()

print("[IMPORT_TEST] Calling app.close()", flush=True)
app.close()
print("[IMPORT_TEST] App closed, exiting", flush=True)
sys.stdout.flush()
