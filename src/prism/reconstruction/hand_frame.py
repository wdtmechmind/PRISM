"""Absolute MechHand body frame derived from the LED constellation.

The four LEDs are rigidly mounted on the hand, so their world positions fix the
hand pose outright -- no hand-eye calibration and no caliper measurement of the
LED offsets are needed:

    forward (toward the fingertips) = red    -> blue
    right   (little -> thumb)       = yellow -> green
    up      (back of hand)          = right x forward
    origin  (MechHand ``base_link``, i.e. the arm-flange mount)
                                    = blue + 5mm * forward + 30mm * up

``forward`` is authoritative; ``right`` is orthogonalised against it and the
residual skew is reported so a mis-assembled rig is caught early.

The MechHand is a LEFT hand, so ``right`` runs from the little finger toward the
thumb. Getting that direction backwards flips ``up`` too and silently rotates
every replayed trajectory by 180 degrees about the palm normal, with no change
in ``skew_deg`` -- see :func:`reference_pose_error` for the check that catches it.

Poses built from this definition are absolute and comparable across trials,
unlike the ad-hoc first-frame frame in
:func:`prism.reconstruction.realtime_reconstruction.build_body_model`.
"""

import json
import os

import numpy as np


# Columns are the semantic axes (forward, left, up) expressed in MechHand
# ``base_link``. Derived from MechHand_deg45_V2.urdf: forward is the mean F2..F5
# finger extension direction (MCP->PIP), right is the F5->F2 knuckle direction
# (little -> thumb, this being a left hand).
BASELINK_FROM_SEMANTIC = np.array([
    [-0.563187234, -0.013630801, 0.826216885],
    [0.064567990, -0.997532795, 0.027555351],
    [0.823802837, 0.068865986, 0.562677849],
], dtype=np.float64)

HAND_CHIRALITY = 'left'

# Semantic axis directions expressed in semantic coordinates, where the basis
# columns are ordered (forward, left, up).
SEMANTIC_AXES = {
    'forward': (1.0, 0.0, 0.0),
    'backward': (-1.0, 0.0, 0.0),
    'left': (0.0, 1.0, 0.0),
    'right': (0.0, -1.0, 0.0),
    'up': (0.0, 0.0, 1.0),
    'down': (0.0, 0.0, -1.0),
}

DEFAULT_SPEC = {
    'forward_from': ['red', 'blue'],
    'right_from': ['yellow', 'green'],
    'origin_anchor': 'blue',
    'origin_forward_m': 0.005,
    'origin_right_m': 0.0,
    'origin_up_m': -0.03,
    'baselink_from_semantic': BASELINK_FROM_SEMANTIC.tolist(),
    'max_skew_deg': 15.0,
    # The hand sits at a different angle in the handheld capture rig than on the
    # arm flange; rotating the captured hand by this much reaches the robot
    # installation pose. Applied about the hand's own axis, so it right-multiplies.
    'mount_correction_axis': 'right',
    'mount_correction_deg': -45.0,
}

DEFAULT_SPEC_PATH = os.path.join('configs', 'devices', 'mechhand_led_body.json')


def load_spec(path=None):
    """Return the hand-frame spec, overlaying ``path`` (or the default) if present."""
    spec = dict(DEFAULT_SPEC)
    if path is None:
        path = os.environ.get('PRISM_HAND_FRAME_SPEC', DEFAULT_SPEC_PATH)
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            spec.update(json.load(f) or {})
    return spec


def required_colors(spec=None):
    spec = spec or DEFAULT_SPEC
    names = list(spec['forward_from']) + list(spec['right_from']) + [spec['origin_anchor']]
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _axis_angle(axis, angle_rad):
    a = _unit(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle_rad) * k + (1.0 - np.cos(angle_rad)) * (k @ k)


def mount_correction(spec=None):
    """Rotation taking the capture-rig hand mounting to the arm-flange mounting."""
    spec = spec or DEFAULT_SPEC
    deg = float(spec.get('mount_correction_deg', 0.0))
    if abs(deg) < 1e-12:
        return np.eye(3, dtype=np.float64)
    name = str(spec.get('mount_correction_axis', 'right'))
    if name not in SEMANTIC_AXES:
        raise ValueError('unknown mount_correction_axis: %s' % name)
    return _axis_angle(SEMANTIC_AXES[name], np.radians(deg))


def semantic_frame_from_leds(points_by_color, spec=None):
    """Return ``(R_world_semantic, origin_world, skew_deg)`` for the hand *as captured*.

    Columns of ``R_world_semantic`` are (forward, left, up). This is the raw
    capture-rig mounting, before :func:`mount_correction` is applied.
    """
    spec = spec or DEFAULT_SPEC
    for name in required_colors(spec):
        if name not in points_by_color:
            return None

    def pt(name):
        return np.asarray(points_by_color[name], dtype=np.float64).reshape(3)

    fwd = _unit(pt(spec['forward_from'][1]) - pt(spec['forward_from'][0]))
    right_raw = _unit(pt(spec['right_from'][1]) - pt(spec['right_from'][0]))
    if fwd is None or right_raw is None:
        return None

    skew_deg = 90.0 - float(np.degrees(np.arccos(np.clip(abs(np.dot(fwd, right_raw)), -1.0, 1.0))))
    right = _unit(right_raw - np.dot(right_raw, fwd) * fwd)
    if right is None:
        return None
    up = _unit(np.cross(right, fwd))
    if up is None:
        return None

    origin = (pt(spec['origin_anchor'])
              + float(spec['origin_forward_m']) * fwd
              + float(spec.get('origin_right_m', 0.0)) * right
              + float(spec['origin_up_m']) * up)
    return np.column_stack([fwd, -right, up]), origin, abs(skew_deg)


def hand_pose_from_leds(points_by_color, spec=None):
    """Return the as-captured ``led_base_link`` pose and skew, or ``None``.

    Mechanical arm-mount correction is deliberately not applied here. Recorded
    trajectories remain faithful to the LED rig; deployment applies the mount
    correction when converting the trajectory to a robot command frame.

    ``skew_deg`` is how far the measured red->blue and yellow->green directions
    are from perpendicular; large values mean the colour assignment or the rig
    geometry is wrong.
    """
    spec = spec or DEFAULT_SPEC
    solved = semantic_frame_from_leds(points_by_color, spec)
    if solved is None:
        return None
    rot_world_semantic, origin, skew_deg = solved

    baselink_from_semantic = np.asarray(spec['baselink_from_semantic'], dtype=np.float64).reshape(3, 3)
    rot_world_baselink = rot_world_semantic @ baselink_from_semantic.T
    return rot_world_baselink, origin, skew_deg


def apply_mount_correction(rot_world_led_base, spec=None):
    """Rotate an as-captured LED-base pose into the robot installation pose."""
    spec = spec or DEFAULT_SPEC
    baselink_from_semantic = np.asarray(spec['baselink_from_semantic'], dtype=np.float64).reshape(3, 3)
    correction_in_led_base = baselink_from_semantic @ mount_correction(spec) @ baselink_from_semantic.T
    return np.asarray(rot_world_led_base, dtype=np.float64).reshape(3, 3) @ correction_in_led_base


def reference_pose_error(points_by_color, expected_up_world=(0.0, 0.0, 1.0), spec=None):
    """Angle in degrees between the LED-derived back-of-hand normal and ``expected_up_world``.

    Capture the hand resting palm-down on a flat surface and call this: the
    result should be near 0. Near 180 means the LED direction spec is reversed
    (``right_from`` or ``forward_from`` swapped), which ``skew_deg`` cannot see
    because the two axes stay perpendicular either way.

    Uses the as-captured frame, so it is unaffected by
    :func:`mount_correction`. It validates the LED spec only:
    ``baselink_from_semantic`` cancels out and cannot be checked against
    captured data at all -- it is fixed by URDF geometry and verified there.
    """
    spec = spec or DEFAULT_SPEC
    solved = semantic_frame_from_leds(points_by_color, spec)
    if solved is None:
        return None
    up = solved[0][:, 2]
    return float(np.degrees(np.arccos(np.clip(np.dot(up, _unit(expected_up_world)), -1.0, 1.0))))


def build_hand_body_model(points_by_color, spec=None):
    """Build a body model whose frame is the as-captured ``led_base_link``.

    Shape-compatible with
    :func:`prism.reconstruction.realtime_reconstruction.build_body_model`, so
    ``estimate_pose_from_model`` then returns the true hand pose directly.
    """
    spec = spec or DEFAULT_SPEC
    solved = hand_pose_from_leds(points_by_color, spec)
    if solved is None:
        return None
    rot_wb, origin, skew_deg = solved

    model = {}
    for name, p in points_by_color.items():
        pw = np.asarray(p, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(pw)):
            continue
        model[name] = rot_wb.T @ (pw - origin)

    return {
        'model_points': model,
        'init_origin': origin,
        'init_rot_wb': rot_wb,
        'frame_colors': required_colors(spec),
        'frame': 'led_base_link',
        'skew_deg': skew_deg,
    }
