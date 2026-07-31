"""
MechHand_deg45_V2 gesture → joint angle mapping.

URDF joint structure
---------------------
F1 (thumb, 3-DOF):
    F1_J1  base_link  → F1_L1   thumb abduction/rotation
    F1_J2  F1_L1      → F1_L2   thumb MCP flexion
    F1_J3  F1_L2      → F1_L3   thumb IP flexion

F2 (index, 2-DOF):
    F2_J1  base_link  → F2_L1   index MCP flexion
    F2_J2  F2_L1      → F2_L2   index PIP/DIP flexion

F3 (middle, 2-DOF):
    F3_J1  base_link  → F3_L1
    F3_J2  F3_L1      → F3_L2

F4 (ring, 2-DOF):
    F4_J1  base_link  → F4_L1
    F4_J2  F4_L1      → F4_L2

F5 (little, 2-DOF):
    F5_J1  base_link  → F5_L1
    F5_J2  F5_L1      → F5_L2

Sign convention (initial assumption — verify in Isaac Sim):
    Positive values = finger flexion (closing toward palm).
    All values in radians.

These values are INITIAL GUESSES derived from the URDF structure.
Run  tools/tune_hand_gestures.py  in Isaac Sim to visually adjust
and save the final calibrated config back to this file.
"""

import math

# ──────────────────────────────────────────────────────────────────
# Physical range estimates (radians) — URDF exports ±π as placeholder
# Typical robot finger MCP: 0 → ~1.57 rad  (0° → 90°)
# Typical robot finger PIP: 0 → ~1.40 rad  (0° → 80°)
# Thumb rotation (F1_J1):  -0.3 → 0.8 rad  (varies per design)
# ──────────────────────────────────────────────────────────────────

# Shorthand helpers
_OPEN  = 0.0
_MCP   = 1.40   # proximal joint fully closed (~80°)
_PIP   = 1.10   # distal  joint fully closed (~63°)
_T_ROT = 0.50   # thumb base rotation when opposing fingers
_T_MCP = 1.10
_T_IP  = 0.80

ALL_JOINT_NAMES = [
    'F1_J1', 'F1_J2', 'F1_J3',   # thumb
    'F2_J1', 'F2_J2',             # index
    'F3_J1', 'F3_J2',             # middle
    'F4_J1', 'F4_J2',             # ring
    'F5_J1', 'F5_J2',             # little
]


def _cfg(f1j1=_OPEN, f1j2=_OPEN, f1j3=_OPEN,
         f2j1=_OPEN, f2j2=_OPEN,
         f3j1=_OPEN, f3j2=_OPEN,
         f4j1=_OPEN, f4j2=_OPEN,
         f5j1=_OPEN, f5j2=_OPEN):
    """Return an ordered dict of all 11 joint angles."""
    return {
        'F1_J1': f1j1, 'F1_J2': f1j2, 'F1_J3': f1j3,
        'F2_J1': f2j1, 'F2_J2': f2j2,
        'F3_J1': f3j1, 'F3_J2': f3j2,
        'F4_J1': f4j1, 'F4_J2': f4j2,
        'F5_J1': f5j1, 'F5_J2': f5j2,
    }


# ──────────────────────────────────────────────────────────────────
# Gesture configs  (keyed by pose name from socket_client.py)
# ──────────────────────────────────────────────────────────────────

GESTURE_JOINT_CONFIGS = {
    # [01] Five-finger grasp
    'five_grasp': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f2j1=1.4000, f2j2=1.1000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [02] Five-finger open
    'five_open': _cfg(f1j1=0.5500),

    # [03] Two-finger grasp (A)
    'two_grasp_a': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f2j1=1.4000, f2j2=1.1000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [04] Two-finger open (A)
    'two_open_a': _cfg(f1j1=0.5000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [05] Two-finger grasp (B)
    'two_grasp_b': _cfg(f1j1=0.5000, f1j2=0.8400, f1j3=0.7900, f2j1=1.4000, f2j2=1.1000),

    # [06] Two-finger open (B)
    'two_open_b': _cfg(f1j1=0.5000),

    # [07] Three-finger grasp (A)
    'three_grasp_a': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f2j1=1.4000, f2j2=1.1000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [08] Three-finger open (A)
    'three_open_a': _cfg(f1j1=0.5000),

    # [09] Three-finger grasp (B)
    'three_grasp_b': _cfg(f1j1=0.7100, f1j2=0.9000, f1j3=0.8000, f2j1=1.4000, f2j2=1.1000, f3j1=1.4000, f3j2=1.1000),

    # [10] Three-finger open (B)
    'three_open_b': _cfg(),

    # [11] Five-finger sequence motion
    'five_sequence': _cfg(f1j1=-0.1600, f1j2=0.9200, f1j3=0.9000, f2j1=1.5800, f2j2=1.1300, f3j1=1.5600, f3j2=1.2100, f4j1=1.6600, f4j2=1.2400, f5j1=1.6900, f5j2=1.2700),

    # [12] Thumb inward
    'thumb_in': _cfg(f1j1=-0.5500, f1j3=-0.0800),

    # [13] Thumb outward
    'thumb_out': _cfg(f1j1=0.5500),

    # [14] Index pointing
    'index_point': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [15] Index press
    'index_press': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f2j1=-0.0300, f2j2=0.6000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [16] Index single click
    'index_single_click': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

    # [17] Index double click
    'index_double_click': _cfg(f1j1=0.5000, f1j2=1.1000, f1j3=0.8000, f3j1=1.4000, f3j2=1.1000, f4j1=1.4000, f4j2=1.1000, f5j1=1.4000, f5j2=1.1000),

}

GESTURE_JOINT_CONFIGS['grasp']        = GESTURE_JOINT_CONFIGS['five_grasp']
GESTURE_JOINT_CONFIGS['open']         = GESTURE_JOINT_CONFIGS['five_open']
GESTURE_JOINT_CONFIGS['three_grasp']  = GESTURE_JOINT_CONFIGS['three_grasp_a']
GESTURE_JOINT_CONFIGS['index_click']  = GESTURE_JOINT_CONFIGS['index_single_click']


def get_joint_config(pose_name):
    """Return joint angle dict for *pose_name*, or None if unknown."""
    return GESTURE_JOINT_CONFIGS.get(pose_name)


def interpolate_configs(cfg_a, cfg_b, t):
    """Linear interpolation between two joint configs.  t in [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return {k: cfg_a[k] + (cfg_b[k] - cfg_a[k]) * t for k in ALL_JOINT_NAMES}
