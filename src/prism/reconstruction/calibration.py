import json
import os

import numpy as np


def _normalize_vec(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _rotation_from_to(a, b):
    a = _normalize_vec(a)
    b = _normalize_vec(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize_vec(np.cross(a, axis))
        x, y, z = axis
        return np.array([
            [2 * x * x - 1, 2 * x * y, 2 * x * z],
            [2 * x * y, 2 * y * y - 1, 2 * y * z],
            [2 * x * z, 2 * y * z, 2 * z * z - 1],
        ], dtype=np.float64)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def apply_corrected_transform(points_xyz, corrected_transform):
    pts = np.asarray(points_xyz, dtype=np.float64)
    out = np.full_like(pts, np.nan, dtype=np.float64)
    if pts.size == 0:
        return out
    finite = np.isfinite(pts).all(axis=1)
    if not np.any(finite):
        return out

    origin = corrected_transform['origin']
    rot = corrected_transform['R']
    out[finite] = ((pts[finite] - origin) @ rot.T)
    return out


def load_calibration(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intr = data['intrinsics']
    extr = data['multi_camera']['extrinsics']

    cameras = {
        0: {
            'K': np.asarray(intr['cam0']['K'], dtype=np.float64),
            'D': np.asarray(intr['cam0']['D'], dtype=np.float64),
            'R': np.eye(3, dtype=np.float64),
            't': np.zeros((3,), dtype=np.float64),
        }
    }

    for i in [1, 2, 3]:
        key = 'cam0_to_cam%d' % i
        if key not in extr:
            raise RuntimeError('missing extrinsic key: %s' % key)
        cameras[i] = {
            'K': np.asarray(intr['cam%d' % i]['K'], dtype=np.float64),
            'D': np.asarray(intr['cam%d' % i]['D'], dtype=np.float64),
            'R': np.asarray(extr[key]['R'], dtype=np.float64),
            't': np.asarray(extr[key]['t'], dtype=np.float64).reshape(3),
        }

    return cameras


def load_calibration_serial_map(path):
    """Return mapping: logical cam index -> physical camera serial string.

    The mapping is extracted from each intrinsic folder path when available,
    e.g. ".../cam1_DA8165486" -> {1: "DA8165486"}.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intr = data.get('intrinsics', {})
    serial_map = {}
    for i in [0, 1, 2, 3]:
        key = 'cam%d' % i
        folder = str((intr.get(key) or {}).get('folder', '') or '')
        serial = ''
        if folder:
            base = os.path.basename(folder.rstrip('/'))
            prefix = '%s_' % key
            if base.startswith(prefix):
                serial = base[len(prefix):].strip()
        serial_map[i] = serial
    return serial_map


def get_camera_centers_world(cameras):
    centers = {}
    for cam_idx, cam in cameras.items():
        r = cam['R']
        t = cam['t'].reshape(3, 1)
        c = -r.T @ t
        centers[cam_idx] = c.reshape(3)
    return centers


def build_corrected_transform(cameras, camera_centers):
    if len(camera_centers) < 3:
        return {'origin': np.zeros(3, dtype=np.float64), 'R': np.eye(3, dtype=np.float64)}

    cam_ids = sorted(camera_centers.keys())
    arr = np.asarray([camera_centers[i] for i in cam_ids], dtype=np.float64)
    plane_center = np.mean(arr, axis=0)

    demean = arr - plane_center
    _, _, vh = np.linalg.svd(demean, full_matrices=False)
    normal = _normalize_vec(vh[-1, :])

    fwd = np.zeros(3, dtype=np.float64)
    for cam_idx in cam_ids:
        r = cameras[cam_idx]['R']
        fwd += (r.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
    fwd = _normalize_vec(fwd)

    if float(np.dot(normal, fwd)) > 0.0:
        normal = -normal

    rot_plane = _rotation_from_to(normal, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    def _project_to_plane(point):
        point = np.asarray(point, dtype=np.float64).reshape(3)
        return point - normal * float(np.dot(point - plane_center, normal))

    def _line_intersection_2d(a0, a1, b0, b1):
        a0 = np.asarray(a0, dtype=np.float64).reshape(2)
        a1 = np.asarray(a1, dtype=np.float64).reshape(2)
        b0 = np.asarray(b0, dtype=np.float64).reshape(2)
        b1 = np.asarray(b1, dtype=np.float64).reshape(2)
        da = a1 - a0
        db = b1 - b0
        denom = da[0] * db[1] - da[1] * db[0]
        if abs(float(denom)) < 1e-12:
            return None
        diff = b0 - a0
        t = (diff[0] * db[1] - diff[1] * db[0]) / denom
        return a0 + t * da

    projected = {cid: _project_to_plane(camera_centers[cid]) for cid in cam_ids}
    origin = np.asarray(plane_center, dtype=np.float64)
    if all(cid in projected for cid in [0, 1, 2, 3]):
        q0 = (projected[0] - plane_center) @ rot_plane.T
        q1 = (projected[1] - plane_center) @ rot_plane.T
        q2 = (projected[2] - plane_center) @ rot_plane.T
        q3 = (projected[3] - plane_center) @ rot_plane.T
        qx = _line_intersection_2d(q0[:2], q2[:2], q1[:2], q3[:2])
        if qx is not None and np.isfinite(qx).all():
            origin = plane_center + np.array([qx[0], qx[1], 0.0], dtype=np.float64) @ rot_plane

    c0 = np.asarray(projected.get(0, arr[0]), dtype=np.float64)
    c1 = projected.get(1, None)
    if c1 is None:
        # Fallback: choose the first available non-cam0 center.
        fallback_ids = [cid for cid in cam_ids if cid != 0]
        c1 = projected.get(fallback_ids[0], arr[-1]) if fallback_ids else arr[-1]
    c1 = np.asarray(c1, dtype=np.float64)
    v = (c1 - c0) @ rot_plane.T
    vx, vy = float(v[0]), float(v[1])
    theta = np.arctan2(vy, vx)
    ct = float(np.cos(-theta))
    st = float(np.sin(-theta))
    rot_z = np.array([
        [ct, -st, 0.0],
        [st, ct, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    rot = rot_z @ rot_plane
    return {'origin': origin, 'R': rot}
