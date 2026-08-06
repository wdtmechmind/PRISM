#!/usr/bin/env python3
"""Interactive Isaac Sim tool for tuning MechHand gesture joint angles.

Loads the MechHand URDF, shows a slider panel for each joint, and lets
you step through every gesture to visually verify and adjust angles.
Saves the final values back to src/prism/devices/hand/joint_config.py.

Usage:
    /isaac-sim/python.sh tools/tune_hand_gestures.py \
        --urdf MechHand_deg45_V2_URDF/MechHand_deg45_V2.urdf

Controls:
    ◀ / ▶ buttons    Previous / next gesture
    sliders          Drag to adjust joint angles in real-time
    Save gesture     Store current slider values for the active gesture
    Write file       Persist all saved gestures to joint_config.py and exit
    Reset            Revert active gesture to joint_config.py defaults
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# ── repo path setup ───────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from prism.devices.hand.socket_client import GESTURE_TABLE  # noqa: E402
from prism.devices.hand.joint_config import (  # noqa: E402
    ALL_JOINT_NAMES,
    GESTURE_JOINT_CONFIGS,
    get_joint_config,
)

# ── Isaac Sim bootstrap ───────────────────────────────────────────
from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({'headless': False, 'renderer': 'RayTracedLighting'})

import omni.kit.commands  # noqa: E402
import omni.timeline       # noqa: E402
import omni.usd            # noqa: E402
import omni.ui as ui       # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux, UsdPhysics  # noqa: E402

# ── constants ─────────────────────────────────────────────────────
# Gesture list in SDK order (id, pose_name, display_name)
GESTURES = [(gid, pose, name) for gid, pose, name in GESTURE_TABLE]

_SIGN_NOTE = 'Positive = flexion (closing). Verify sign in Isaac Sim; flip in joint_config.py if needed.'


# ─────────────────────────────────────────────────────────────────
class HandTuner:
    def __init__(self, urdf_path: str):
        self._urdf_path = os.path.abspath(urdf_path)
        self._articulation: SingleArticulation | None = None
        self._prim_path: str = ''
        self._gesture_idx = 0
        # Working copy — edits here, flushed to file on 'Write file'
        self._configs: dict[str, dict[str, float]] = {
            pose: dict(cfg) for pose, cfg in GESTURE_JOINT_CONFIGS.items()
        }
        self._sliders: dict[str, ui.FloatSlider] = {}
        self._label_gesture: ui.Label | None = None
        self._setup_scene()
        self._build_ui()

    # ── scene ──────────────────────────────────────────────────────
    def _setup_scene(self):
        # Patch 'package:///meshes/...' to absolute paths so Isaac Sim can load STLs.
        # SolidWorks URDF exporter writes package:// URIs; Isaac Sim can't resolve them.
        urdf_dir = os.path.dirname(self._urdf_path)
        meshes_dir = os.path.join(urdf_dir, 'meshes')
        patched_urdf = self._urdf_path + '.patched.urdf'
        with open(self._urdf_path, 'r', encoding='utf-8') as f:
            src = f.read()
        src = src.replace('package:///meshes/', meshes_dir + '/')
        with open(patched_urdf, 'w', encoding='utf-8') as f:
            f.write(src)
        print('[scene] Patched URDF written to: %s' % patched_urdf)

        # Create import config via kit commands
        status, import_config = omni.kit.commands.execute('URDFCreateImportConfig')
        if not status:
            raise RuntimeError('URDFCreateImportConfig failed')
        import_config.merge_fixed_joints = False
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.import_inertia_tensor = True
        import_config.create_physics_scene = False  # we create it below
        # Position drives so sliders move joints directly
        import_config.set_default_drive_type(1)           # 1 = position
        import_config.set_default_drive_strength(1e6)
        import_config.set_default_position_drive_damping(1e4)

        status, self._prim_path = omni.kit.commands.execute(
            'URDFParseAndImportFile',
            urdf_path=patched_urdf,
            import_config=import_config,
            get_articulation_root=True,
        )
        if not status or not self._prim_path:
            raise RuntimeError('URDFParseAndImportFile failed for: %s' % patched_urdf)
        print('[scene] Imported to prim path: %s' % self._prim_path)

        # Physics scene
        stage = omni.usd.get_context().get_stage()
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path('/physicsScene'))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

        # Lighting
        UsdLux.DistantLight.Define(stage, Sdf.Path('/DistantLight')).CreateIntensityAttr(600)

        # Start simulation (required before Articulation.initialize())
        omni.timeline.get_timeline_interface().play()
        _app.update()
        _app.update()

        self._articulation = SingleArticulation(prim_path=self._prim_path)
        self._articulation.initialize()

        # Verify joint names match
        dof_names = list(self._articulation.dof_names)
        missing = [j for j in ALL_JOINT_NAMES if j not in dof_names]
        if missing:
            print('[WARN] Joints in joint_config but not in articulation: %s' % missing)
        print('[scene] Articulation DOFs: %s' % dof_names)

    # ── UI ─────────────────────────────────────────────────────────
    def _build_ui(self):
        self._window = ui.Window('Hand Gesture Tuner', width=420, height=700)
        with self._window.frame:
            with ui.VStack(spacing=4):
                # ── gesture header ──
                with ui.HStack(height=36):
                    ui.Button('◀  Prev', clicked_fn=self._prev_gesture, width=100)
                    self._label_gesture = ui.Label(
                        self._gesture_label(), alignment=ui.Alignment.CENTER
                    )
                    ui.Button('Next  ▶', clicked_fn=self._next_gesture, width=100)

                ui.Label(_SIGN_NOTE, word_wrap=True, height=24,
                         style={'color': 0xFFAAAAAA, 'font_size': 11})
                ui.Separator(height=4)

                # ── per-joint sliders ──
                for jname in ALL_JOINT_NAMES:
                    with ui.HStack(height=22):
                        ui.Label(jname, width=70)
                        slider = ui.FloatSlider(
                            min=-3.14, max=3.14,
                            step=0.01,
                            width=250,
                        )
                        slider.model.add_value_changed_fn(
                            lambda m, j=jname: self._on_slider(j, m.get_value_as_float())
                        )
                        self._sliders[jname] = slider

                ui.Separator(height=6)

                # ── action buttons ──
                with ui.HStack(height=32):
                    ui.Button('Reset (r)', clicked_fn=self._reset_gesture, width=100)
                    ui.Button('Save gesture (s)', clicked_fn=self._save_gesture, width=130)
                    ui.Button('Write file (w)', clicked_fn=self._write_and_quit, width=130)

        self._refresh_sliders()

    def _gesture_label(self) -> str:
        gid, pose, display = GESTURES[self._gesture_idx]
        return f'[{gid:02d}]  {pose}  —  {display}'

    # ── navigation ─────────────────────────────────────────────────
    def _next_gesture(self):
        self._gesture_idx = (self._gesture_idx + 1) % len(GESTURES)
        self._refresh_sliders()

    def _prev_gesture(self):
        self._gesture_idx = (self._gesture_idx - 1) % len(GESTURES)
        self._refresh_sliders()

    def _current_pose_name(self) -> str:
        return GESTURES[self._gesture_idx][1]

    # ── slider sync ────────────────────────────────────────────────
    def _refresh_sliders(self):
        pose = self._current_pose_name()
        if self._label_gesture:
            self._label_gesture.text = self._gesture_label()
        cfg = self._configs.get(pose, {j: 0.0 for j in ALL_JOINT_NAMES})
        for jname, slider in self._sliders.items():
            slider.model.set_value(float(cfg.get(jname, 0.0)))
        self._apply_to_hand(cfg)

    def _on_slider(self, joint_name: str, value: float):
        """Called on every slider drag — update the hand in real-time."""
        pose = self._current_pose_name()
        if pose not in self._configs:
            self._configs[pose] = {j: 0.0 for j in ALL_JOINT_NAMES}
        self._configs[pose][joint_name] = value
        self._apply_to_hand(self._configs[pose])

    # ── articulation drive ─────────────────────────────────────────
    def _apply_to_hand(self, cfg: dict[str, float]):
        if self._articulation is None:
            return
        dof_names = list(self._articulation.dof_names)
        positions = np.array(self._articulation.get_joint_positions(), dtype=np.float64)
        for jname, angle in cfg.items():
            if jname in dof_names:
                positions[dof_names.index(jname)] = float(angle)
        self._articulation.set_joint_positions(positions)

    # ── save / reset ───────────────────────────────────────────────
    def _save_gesture(self):
        pose = self._current_pose_name()
        print('[save] %s saved.' % pose)

    def _reset_gesture(self):
        pose = self._current_pose_name()
        original = GESTURE_JOINT_CONFIGS.get(pose, {j: 0.0 for j in ALL_JOINT_NAMES})
        self._configs[pose] = dict(original)
        self._refresh_sliders()

    # ── write to joint_config.py ───────────────────────────────────
    def _write_and_quit(self):
        out_path = os.path.join(_SRC_DIR, 'prism', 'devices', 'hand', 'joint_config.py')
        _write_joint_config(out_path, self._configs)
        print('[write] Saved to %s' % out_path)
        _app.close()

    # ── main loop ─────────────────────────────────────────────────
    def run(self):
        while _app.is_running():
            _app.update()


# ─────────────────────────────────────────────────────────────────
def _fmt_cfg(cfg: dict[str, float]) -> str:
    """Format a single gesture config as a _cfg(...) call."""
    args = []
    name_map = {
        'F1_J1': 'f1j1', 'F1_J2': 'f1j2', 'F1_J3': 'f1j3',
        'F2_J1': 'f2j1', 'F2_J2': 'f2j2',
        'F3_J1': 'f3j1', 'F3_J2': 'f3j2',
        'F4_J1': 'f4j1', 'F4_J2': 'f4j2',
        'F5_J1': 'f5j1', 'F5_J2': 'f5j2',
    }
    non_zero = {name_map[k]: v for k, v in cfg.items() if abs(v) > 1e-6}
    if not non_zero:
        return '_cfg()'
    parts = ', '.join('%s=%.4f' % (k, v) for k, v in non_zero.items())
    return '_cfg(%s)' % parts


def _write_joint_config(out_path: str, configs: dict[str, dict[str, float]]):
    """Rewrite the GESTURE_JOINT_CONFIGS block inside joint_config.py."""
    with open(out_path, 'r', encoding='utf-8') as f:
        src = f.read()

    # Build replacement block
    lines = ['GESTURE_JOINT_CONFIGS = {\n']
    # Write in SDK gesture order first
    written = set()
    for gid, pose, display in GESTURE_TABLE:
        if pose in configs:
            lines.append('    # [%02d] %s\n' % (gid, display))
            lines.append('    %r: %s,\n\n' % (pose, _fmt_cfg(configs[pose])))
            written.add(pose)
    # Then aliases and any extras
    lines.append('}\n')
    # Append alias lines
    alias_block = (
        "\nGESTURE_JOINT_CONFIGS['grasp']        = GESTURE_JOINT_CONFIGS['five_grasp']\n"
        "GESTURE_JOINT_CONFIGS['open']         = GESTURE_JOINT_CONFIGS['five_open']\n"
        "GESTURE_JOINT_CONFIGS['three_grasp']  = GESTURE_JOINT_CONFIGS['three_grasp_a']\n"
        "GESTURE_JOINT_CONFIGS['index_click']  = GESTURE_JOINT_CONFIGS['index_single_click']\n"
    )
    new_block = ''.join(lines) + alias_block

    # Replace old block between markers
    start_marker = 'GESTURE_JOINT_CONFIGS = {'
    end_marker = "GESTURE_JOINT_CONFIGS['index_click']"
    start = src.find(start_marker)
    end = src.find(end_marker)
    if start == -1 or end == -1:
        # Fallback: append
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write('\n# --- tuner output ---\n' + new_block)
    else:
        end = src.find('\n', end) + 1
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(src[:start] + new_block + src[end:])


# ─────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--urdf',
        default=os.path.join(_REPO_ROOT, 'MechHand_deg45_V2_URDF', 'MechHand_deg45_V2.urdf'),
        help='Path to MechHand URDF file',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    tuner = HandTuner(urdf_path=args.urdf)
    tuner.run()
