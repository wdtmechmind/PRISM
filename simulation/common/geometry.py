"""Geometry helpers used by the simulation preprocessing pipeline."""

from __future__ import annotations

import math

import numpy as np


def rotation_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll) for degree inputs."""
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def mat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def mat_to_euler_zyx_deg(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to roll, pitch, yaw degrees using ZYX convention."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    sy = -float(m[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-9:
        roll = math.atan2(float(m[2, 1]), float(m[2, 2]))
        yaw = math.atan2(float(m[1, 0]), float(m[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(m[0, 1]), float(m[1, 1]))
    return np.degrees([roll, pitch, yaw]).astype(np.float64)
