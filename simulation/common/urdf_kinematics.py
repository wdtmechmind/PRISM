"""Small URDF kinematics helpers for offline simulation planning."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = a / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def transform_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return transform


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    rot = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    cos_angle = max(-1.0, min(1.0, 0.5 * (float(np.trace(rot)) - 1.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - angle) < 1e-5:
        axis = np.array([
            math.sqrt(max(0.0, (rot[0, 0] + 1.0) * 0.5)),
            math.sqrt(max(0.0, (rot[1, 1] + 1.0) * 0.5)),
            math.sqrt(max(0.0, (rot[2, 2] + 1.0) * 0.5)),
        ], dtype=np.float64)
        axis[0] = math.copysign(axis[0], rot[2, 1] - rot[1, 2])
        axis[1] = math.copysign(axis[1], rot[0, 2] - rot[2, 0])
        axis[2] = math.copysign(axis[2], rot[1, 0] - rot[0, 1])
        norm = max(float(np.linalg.norm(axis)), 1e-12)
        return axis / norm * angle
    axis = np.array([
        rot[2, 1] - rot[1, 2],
        rot[0, 2] - rot[2, 0],
        rot[1, 0] - rot[0, 1],
    ], dtype=np.float64) / (2.0 * math.sin(angle))
    return axis * angle


def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(position: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


class SerialChain:
    def __init__(self, joints: Sequence[UrdfJoint], active_joint_names: Sequence[str]):
        self.joints = list(joints)
        self.active_joint_names = list(active_joint_names)
        self.active_indices = [i for i, joint in enumerate(self.joints) if joint.name in self.active_joint_names]
        self.lowers = np.array([self.joints[i].lower for i in self.active_indices], dtype=np.float64)
        self.uppers = np.array([self.joints[i].upper for i in self.active_indices], dtype=np.float64)
        self.fixed_transforms = [transform_matrix(joint.xyz, joint.rpy) for joint in self.joints]
        self.active_columns = {joint_index: col for col, joint_index in enumerate(self.active_indices)}

    def fk(self, q: Sequence[float]) -> np.ndarray:
        q_arr = np.asarray(q, dtype=np.float64).reshape(len(self.active_indices))
        q_by_joint = {self.joints[index].name: q_arr[i] for i, index in enumerate(self.active_indices)}
        transform = np.eye(4, dtype=np.float64)
        for joint in self.joints:
            transform = transform @ transform_matrix(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = axis_angle_to_matrix(joint.axis, q_by_joint.get(joint.name, 0.0))
                transform = transform @ motion
            elif joint.joint_type == "prismatic":
                motion = np.eye(4, dtype=np.float64)
                motion[:3, 3] = joint.axis * q_by_joint.get(joint.name, 0.0)
                transform = transform @ motion
        return transform

    def pose_error(self, q: Sequence[float], target: np.ndarray, orientation_weight: float) -> np.ndarray:
        current = self.fk(q)
        pos_err = target[:3, 3] - current[:3, 3]
        if orientation_weight <= 0.0:
            return pos_err
        rot_err = rotation_matrix_to_rotvec(target[:3, :3] @ current[:3, :3].T)
        return np.concatenate([pos_err, rot_err * float(orientation_weight)])

    def fk_and_error_jacobian(self, q: Sequence[float], target: np.ndarray, orientation_weight: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_arr = np.asarray(q, dtype=np.float64).reshape(len(self.active_indices))
        transform = np.eye(4, dtype=np.float64)
        joint_origins: Dict[int, np.ndarray] = {}
        joint_axes: Dict[int, np.ndarray] = {}
        joint_types: Dict[int, str] = {}
        for joint_index, joint in enumerate(self.joints):
            transform = transform @ self.fixed_transforms[joint_index]
            column = self.active_columns.get(joint_index)
            joint_value = q_arr[column] if column is not None else 0.0
            if joint.joint_type in {"revolute", "continuous"}:
                if column is not None:
                    joint_origins[joint_index] = transform[:3, 3].copy()
                    joint_axes[joint_index] = transform[:3, :3] @ joint.axis
                    joint_types[joint_index] = joint.joint_type
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = axis_angle_to_matrix(joint.axis, joint_value)
                transform = transform @ motion
            elif joint.joint_type == "prismatic":
                if column is not None:
                    joint_origins[joint_index] = transform[:3, 3].copy()
                    joint_axes[joint_index] = transform[:3, :3] @ joint.axis
                    joint_types[joint_index] = joint.joint_type
                motion = np.eye(4, dtype=np.float64)
                motion[:3, 3] = joint.axis * joint_value
                transform = transform @ motion

        pos_err = target[:3, 3] - transform[:3, 3]
        if orientation_weight <= 0.0:
            jac = np.zeros((3, q_arr.size), dtype=np.float64)
        else:
            rot_err = rotation_matrix_to_rotvec(target[:3, :3] @ transform[:3, :3].T)
            jac = np.zeros((6, q_arr.size), dtype=np.float64)
        tip_pos = transform[:3, 3]
        for joint_index, column in self.active_columns.items():
            axis = joint_axes[joint_index]
            norm = float(np.linalg.norm(axis))
            if norm > 1e-12:
                axis = axis / norm
            if joint_types[joint_index] == "prismatic":
                pos_jac = axis
                rot_jac = np.zeros(3, dtype=np.float64)
            else:
                pos_jac = np.cross(axis, tip_pos - joint_origins[joint_index])
                rot_jac = axis
            jac[:3, column] = -pos_jac
            if orientation_weight > 0.0:
                jac[3:, column] = -rot_jac * float(orientation_weight)
        if orientation_weight <= 0.0:
            err = pos_err
        else:
            err = np.concatenate([pos_err, rot_err * float(orientation_weight)])
        return transform, err, jac

    def numerical_jacobian(self, q: np.ndarray, target: np.ndarray, orientation_weight: float, eps: float = 1e-5) -> np.ndarray:
        base_err = self.pose_error(q, target, orientation_weight)
        jac = np.zeros((base_err.size, q.size), dtype=np.float64)
        for index in range(q.size):
            q_step = q.copy()
            q_step[index] += eps
            jac[:, index] = (self.pose_error(q_step, target, orientation_weight) - base_err) / eps
        return jac

    def solve_ik(
        self,
        target: np.ndarray,
        seed: Sequence[float],
        orientation_weight: float = 0.35,
        max_iters: int = 80,
        tolerance: float = 0.005,
        damping: float = 0.03,
        max_step: float = 0.20,
    ) -> Tuple[np.ndarray, bool, float, float, int]:
        q = np.asarray(seed, dtype=np.float64).reshape(len(self.active_indices)).copy()
        q = np.clip(q, self.lowers, self.uppers)
        pos_error = float("inf")
        rot_error = 0.0
        for iteration in range(int(max_iters)):
            current, err, jac = self.fk_and_error_jacobian(q, target, orientation_weight)
            pos_error = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
            rot_error = float(np.linalg.norm(rotation_matrix_to_rotvec(target[:3, :3] @ current[:3, :3].T)))
            if pos_error <= tolerance and (orientation_weight <= 0.0 or rot_error <= 0.15):
                return q, True, pos_error, rot_error, iteration
            lhs = jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0], dtype=np.float64)
            try:
                dq = -jac.T @ np.linalg.solve(lhs, err)
            except np.linalg.LinAlgError:
                dq = -jac.T @ np.linalg.pinv(lhs) @ err
            norm = float(np.linalg.norm(dq))
            if norm > max_step:
                dq *= max_step / norm
            q = np.clip(q + dq, self.lowers, self.uppers)
        current = self.fk(q)
        pos_error = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
        rot_error = float(np.linalg.norm(rotation_matrix_to_rotvec(target[:3, :3] @ current[:3, :3].T)))
        return q, False, pos_error, rot_error, int(max_iters)


def _parse_vec(text: str, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(part) for part in text.split()], dtype=np.float64)


def _parse_joint(element: ET.Element) -> UrdfJoint:
    origin = element.find("origin")
    axis = element.find("axis")
    limit = element.find("limit")
    parent = element.find("parent")
    child = element.find("child")
    if parent is None or child is None:
        raise ValueError("joint %s missing parent/child" % element.attrib.get("name", ""))
    lower = -math.pi
    upper = math.pi
    if limit is not None:
        lower = float(limit.attrib.get("lower", lower))
        upper = float(limit.attrib.get("upper", upper))
    if element.attrib.get("type") == "continuous":
        lower = -math.pi
        upper = math.pi
    return UrdfJoint(
        name=element.attrib["name"],
        joint_type=element.attrib.get("type", "fixed"),
        parent=parent.attrib["link"],
        child=child.attrib["link"],
        xyz=_parse_vec(origin.attrib.get("xyz", "") if origin is not None else "", [0.0, 0.0, 0.0]),
        rpy=_parse_vec(origin.attrib.get("rpy", "") if origin is not None else "", [0.0, 0.0, 0.0]),
        axis=_parse_vec(axis.attrib.get("xyz", "") if axis is not None else "", [0.0, 0.0, 1.0]),
        lower=lower,
        upper=upper,
    )


def load_serial_chain(urdf_path: str, base_link: str, tip_link: str, active_joint_names: Sequence[str]) -> SerialChain:
    robot = ET.parse(Path(urdf_path)).getroot()
    joints = [_parse_joint(element) for element in robot.findall("joint")]
    by_parent: Dict[str, List[UrdfJoint]] = {}
    for joint in joints:
        by_parent.setdefault(joint.parent, []).append(joint)

    path: List[UrdfJoint] = []

    def dfs(link: str, current: List[UrdfJoint]) -> bool:
        if link == tip_link:
            path.extend(current)
            return True
        for joint in by_parent.get(link, []):
            if dfs(joint.child, current + [joint]):
                return True
        return False

    if not dfs(base_link, []):
        raise RuntimeError("no URDF chain from %s to %s" % (base_link, tip_link))
    active = [name for name in active_joint_names if any(joint.name == name for joint in path)]
    if len(active) != len(active_joint_names):
        missing = [name for name in active_joint_names if name not in active]
        raise RuntimeError("active joints not found in chain: %s" % ", ".join(missing))
    return SerialChain(path, active_joint_names)
