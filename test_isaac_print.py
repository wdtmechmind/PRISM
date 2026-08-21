#!/usr/bin/env python3
"""Test if print statements work in Isaac Sim."""
import sys
import sys

print("STDOUT: Test 1 - Before Isaac Sim", flush=True)
sys.stderr.write("STDERR: Test 2 - Before Isaac Sim\n")
sys.stderr.flush()

from isaacsim import SimulationApp

print("STDOUT: Test 3 - After import", flush=True)
sys.stderr.write("STDERR: Test 4 - After import\n")
sys.stderr.flush()

simulation_app = SimulationApp({"headless": True})

print("STDOUT: Test 5 - After SimulationApp", flush=True)
sys.stderr.write("STDERR: Test 6 - After SimulationApp\n")
sys.stderr.flush()

import omni.usd

print("STDOUT: Test 7 - After omni.usd", flush=True)
sys.stderr.write("STDERR: Test 8 - After omni.usd\n")
sys.stderr.flush()

usd_context = omni.usd.get_context()
usd_context.new_stage()

print("STDOUT: Test 9 - After stage creation", flush=True)
sys.stderr.write("STDERR: Test 10 - After stage creation\n")
sys.stderr.flush()

simulation_app.close()

print("STDOUT: Test 11 - Done", flush=True)
sys.stderr.write("STDERR: Test 12 - Done\n")
sys.stderr.flush()
