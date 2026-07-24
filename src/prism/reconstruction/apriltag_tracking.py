"""
AprilTag rigid-body tracking core.

An alternative to the LED pipeline: instead of coloured LEDs, several AprilTags
(family ``tag36h11``) are fixed to the rigid body being tracked. Because the
tags face in scattered directions, at any instant only a subset is visible, so
the design goal here is that *any single visible tag is enough* to recover the
full 6-DOF body pose, and additional visible tags only improve accuracy.

Coordinate conventions (shared with the LED pipeline):
    * cam0 is the world / reference frame.
    * ``cameras[i] = {K, D, R, t}`` maps a world point into camera ``i``:
          X_cam_i = R_i @ X_world + t_i
      (cam0 has R = I, t = 0).

Two stages:
    1. Rig calibration (``build_tag_rig``): from many frames where tags are
       co-visible, estimate every tag's fixed pose in a common *body* frame
       (the reference tag's frame). Handled via pairwise relative transforms +
       a maximum-spanning-tree so tags that are never directly co-visible are
       still linked through intermediates.
    2. Tracking (``estimate_body_pose``): per frame, each visible tag votes for
       the body pose; votes are robustly fused (outlier-rejected translation
       median + quaternion average).
"""

import os

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 always present in the camera env
    cv2 = None


APRILTAG_36H11_DICT = 'DICT_APRILTAG_36h11'


# ───────────────────────── synchronized frame source ─────────────────────────


def iter_trial_frames(trial_dir, tol_s=0.008):
    """
    Yield time-associated Hik frames for one trial, mirroring the LED offline
    reconstruction's cross-camera association.

    Yields tuples: (frame_index, t_ref, t_trial, {cam_idx: bgr_frame}).
    The reference camera is the lowest-index Hik stream; every other camera is
    advanced to the frame nearest the reference timestamp and only included when
    within ``tol_s`` seconds.
    """
    # Imported lazily so the module can be imported without the full LED stack.
    from prism.processing.offline_reconstruct import (
        discover_hik_streams,
        read_kv_metadata,
    )

    cameras_dir = os.path.join(trial_dir, 'cameras')
    streams = discover_hik_streams(cameras_dir)
    if len(streams) < 1:
        return

    ref_i = min(streams)
    ts_by_cam = {i: streams[i]['ts'] for i in streams}
    caps = {i: cv2.VideoCapture(streams[i]['video']) for i in streams}

    cur, idx = {}, {}
    for i in streams:
        ok, frame = caps[i].read()
        cur[i] = frame if ok else None
        idx[i] = 0

    meta = read_kv_metadata(os.path.join(trial_dir, 'metadata.yaml'))
    trial_start = meta.get('start_wall_time')
    if not isinstance(trial_start, (int, float)):
        trial_start = float(ts_by_cam[ref_i][0])

    ts_ref = ts_by_cam[ref_i]
    j = 0
    try:
        while cur[ref_i] is not None and j < len(ts_ref):
            t_ref = float(ts_ref[j])
            frames = {ref_i: cur[ref_i]}
            for i in streams:
                if i == ref_i:
                    continue
                ts_i = ts_by_cam[i]
                while (idx[i] + 1 < len(ts_i)
                       and abs(ts_i[idx[i] + 1] - t_ref) <= abs(ts_i[idx[i]] - t_ref)):
                    ok, frame = caps[i].read()
                    if not ok:
                        cur[i] = None
                        break
                    cur[i] = frame
                    idx[i] += 1
                if cur[i] is not None and abs(float(ts_i[idx[i]]) - t_ref) <= tol_s:
                    frames[i] = cur[i]
            yield j, t_ref, t_ref - trial_start, frames

            ok, frame = caps[ref_i].read()
            cur[ref_i] = frame if ok else None
            j += 1
    finally:
        for cap in caps.values():
            cap.release()


# ───────────────────────────── detection ─────────────────────────────


def make_detector():
    """Create an OpenCV ArUco detector configured for the tag36h11 family."""
    if cv2 is None:
        raise RuntimeError('OpenCV (cv2) is required for AprilTag detection')
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, APRILTAG_36H11_DICT)
    )
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement markedly improves pose stability.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_tags(img_bgr, detector):
    """
    Detect tag36h11 markers in one image.

    Returns dict: tag_id -> corners (4, 2) float32 in pixel coordinates,
    ordered [top-left, top-right, bottom-right, bottom-left].

    If the same id appears more than once in a single image the id is DROPPED
    for that image (physically ambiguous — duplicate tags cannot be told apart).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    out = {}
    dupes = set()
    if ids is None:
        return out
    for c, i in zip(corners, ids.reshape(-1)):
        tid = int(i)
        if tid in out or tid in dupes:
            dupes.add(tid)
            out.pop(tid, None)
            continue
        out[tid] = c.reshape(4, 2).astype(np.float64)
    return out


def _object_points(tag_size):
    """Square marker corners in the tag frame (z=0), matching detectMarkers order."""
    h = 0.5 * float(tag_size)
    return np.array([
        [-h,  h, 0.0],
        [ h,  h, 0.0],
        [ h, -h, 0.0],
        [-h, -h, 0.0],
    ], dtype=np.float64)


def _solve_tag_pose_in_cam(corners, cam, tag_size):
    """
    solvePnP a single square tag -> (R_marker_to_cam, t_marker_to_cam, reproj_px).
    Returns None if the solve fails.
    """
    objp = _object_points(tag_size)
    flags = getattr(cv2, 'SOLVEPNP_IPPE_SQUARE', cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(objp, corners.reshape(4, 1, 2),
                                  cam['K'], cam['D'], flags=flags)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, cam['K'], cam['D'])
    reproj = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - corners, axis=1)))
    return R, tvec.reshape(3), reproj


# ───────────────────────── rotation / pose helpers ─────────────────────────


def mat_to_quat(R):
    """Rotation matrix -> quaternion [w, x, y, z] (unit)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_mat(q):
    """Quaternion [w, x, y, z] -> rotation matrix."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def average_quaternions(quats, weights=None):
    """Markley eigenvalue quaternion average. quats: (N, 4). Returns unit quat."""
    quats = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    if len(quats) == 1:
        return quats[0] / (np.linalg.norm(quats[0]) + 1e-12)
    if weights is None:
        weights = np.ones(len(quats), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    # Sign-align to the first quaternion so averaging does not cancel.
    ref = quats[0]
    aligned = np.where((quats @ ref)[:, None] < 0.0, -quats, quats)
    M = (aligned * weights[:, None]).T @ aligned
    _, vecs = np.linalg.eigh(M)
    q = vecs[:, -1]
    return q / (np.linalg.norm(q) + 1e-12)


def make_pose(R, t):
    """Assemble a 4x4 homogeneous transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_pose(T):
    """Inverse of a 4x4 rigid transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def average_poses(poses, weights=None):
    """Average a list of 4x4 poses (translation weighted mean + quat average)."""
    poses = list(poses)
    if len(poses) == 1:
        return poses[0].copy()
    if weights is None:
        weights = np.ones(len(poses), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    w = weights / (weights.sum() + 1e-12)
    t = np.sum([w[i] * poses[i][:3, 3] for i in range(len(poses))], axis=0)
    quats = np.array([mat_to_quat(p[:3, :3]) for p in poses])
    R = quat_to_mat(average_quaternions(quats, weights))
    return make_pose(R, t)


# ───────────────────────── per-frame tag world poses ─────────────────────────


def tag_world_poses(frames_by_cam, cameras, detector, tag_size,
                    max_reproj_px=3.0):
    """
    Estimate each tag's 6-DOF pose in the world (cam0) frame for one time instant.

    ``frames_by_cam``: dict cam_idx -> BGR image (already time-associated).

    Returns dict: tag_id -> {
        'T': (4,4) world pose (averaged over the cameras that see it),
        'num_cams': int,
        'cams': [cam_idx, ...],
        'reproj_px': float (max over cameras),
        'cam_spread_mm': float (max pairwise translation disagreement; a large
                         value flags a wrong --tag-size or a mis-detection),
    }
    """
    per_tag_cam = {}  # tag_id -> list of (T_world, weight, cam_idx, reproj)
    for cam_idx, img in frames_by_cam.items():
        if img is None or cam_idx not in cameras:
            continue
        cam = cameras[cam_idx]
        dets = detect_tags(img, detector)
        for tid, corners in dets.items():
            solved = _solve_tag_pose_in_cam(corners, cam, tag_size)
            if solved is None:
                continue
            R_mc, t_mc, reproj = solved
            if reproj > max_reproj_px:
                continue
            # marker->cam (R_mc, t_mc) and world->cam (R_i, t_i) => marker->world
            R_i = cam['R']
            t_i = cam['t'].reshape(3)
            R_world = R_i.T @ R_mc
            t_world = R_i.T @ (t_mc - t_i)
            T_world = make_pose(R_world, t_world)
            weight = 1.0 / (reproj + 0.5)
            per_tag_cam.setdefault(tid, []).append((T_world, weight, cam_idx, reproj))

    out = {}
    for tid, entries in per_tag_cam.items():
        poses = [e[0] for e in entries]
        weights = [e[1] for e in entries]
        cams = [e[2] for e in entries]
        reprojs = [e[3] for e in entries]
        T = average_poses(poses, weights) if len(poses) > 1 else poses[0]
        if len(poses) > 1:
            ts = np.array([p[:3, 3] for p in poses])
            spread = 0.0
            for a in range(len(ts)):
                for b in range(a + 1, len(ts)):
                    spread = max(spread, float(np.linalg.norm(ts[a] - ts[b])))
            spread_mm = spread * 1000.0
        else:
            spread_mm = 0.0
        out[tid] = {
            'T': T,
            'num_cams': len(poses),
            'cams': cams,
            'reproj_px': float(max(reprojs)),
            'cam_spread_mm': spread_mm,
        }
    return out


# ───────────────────────────── rig calibration ─────────────────────────────


def build_tag_rig(frame_tag_poses, reference_tag=None, min_pair_frames=5):
    """
    Build the fixed body model from a sequence of per-frame tag world poses.

    ``frame_tag_poses``: list over frames; each item is the dict returned by
    ``tag_world_poses`` (tag_id -> {'T': ...}).

    Strategy: for every co-visible tag pair (a, b) accumulate the relative
    transform inv(T_a) @ T_b, average it, then run a maximum-spanning-tree from
    the reference tag so every tag receives a body-frame pose even if it was
    never directly co-visible with the reference.

    Returns dict:
        {
          'reference_tag': int,
          'tags': {tag_id: {'T_body_tag': 4x4, 'link_from': int|None,
                            'pair_frames': int, 'trans_std_mm': float}},
          'skipped': [tag_id, ...],   # tags that could not be linked
        }
    """
    # 1. Count visibility and accumulate pairwise relative transforms.
    seen_count = {}
    rel_samples = {}  # (a, b) ordered a<b -> list of inv(Ta)@Tb
    for poses in frame_tag_poses:
        ids = sorted(poses.keys())
        for tid in ids:
            seen_count[tid] = seen_count.get(tid, 0) + 1
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                rel = invert_pose(poses[a]['T']) @ poses[b]['T']
                rel_samples.setdefault((a, b), []).append(rel)

    if not seen_count:
        return {'reference_tag': None, 'tags': {}, 'skipped': []}

    # 2. Average each pair; keep only well-observed edges.
    edges = {}  # (a, b) -> {'T': mean rel, 'n': count, 'trans_std_mm': float}
    for (a, b), samples in rel_samples.items():
        if len(samples) < min_pair_frames:
            continue
        T = average_poses(samples)
        ts = np.array([s[:3, 3] for s in samples])
        trans_std_mm = float(np.mean(np.std(ts, axis=0)) * 1000.0)
        edges[(a, b)] = {'T': T, 'n': len(samples), 'trans_std_mm': trans_std_mm}

    # 3. Pick reference: user choice, else most-seen tag.
    if reference_tag is None or reference_tag not in seen_count:
        reference_tag = max(seen_count, key=lambda k: seen_count[k])

    # Adjacency with relative transform parent->child in both directions.
    adj = {tid: [] for tid in seen_count}
    for (a, b), e in edges.items():
        adj[a].append((b, e['T'], e['n'], e['trans_std_mm']))          # a->b
        adj[b].append((a, invert_pose(e['T']), e['n'], e['trans_std_mm']))  # b->a

    # 4. Maximum-spanning-tree (greedy by edge sample count) from reference.
    tags = {reference_tag: {
        'T_body_tag': np.eye(4, dtype=np.float64),
        'link_from': None,
        'pair_frames': int(seen_count[reference_tag]),
        'trans_std_mm': 0.0,
    }}
    visited = {reference_tag}
    # frontier: list of (count, parent, child, rel_parent_child, std)
    frontier = [(n, reference_tag, c, T, std) for (c, T, n, std) in adj[reference_tag]]
    while frontier:
        frontier.sort(key=lambda x: x[0], reverse=True)
        n, parent, child, rel, std = frontier.pop(0)
        if child in visited:
            continue
        T_body_child = tags[parent]['T_body_tag'] @ rel
        tags[child] = {
            'T_body_tag': T_body_child,
            'link_from': int(parent),
            'pair_frames': int(n),
            'trans_std_mm': float(std),
        }
        visited.add(child)
        for (c, T, cnt, cstd) in adj[child]:
            if c not in visited:
                frontier.append((cnt, child, c, T, cstd))

    skipped = [tid for tid in seen_count if tid not in visited]
    return {'reference_tag': int(reference_tag), 'tags': tags, 'skipped': skipped}


# ───────────────────────────── body pose tracking ─────────────────────────────


def estimate_body_pose(poses, rig, max_trans_dev_m=0.03):
    """
    Fuse all visible tags into one body pose using a pre-calibrated rig.

    ``poses``: dict tag_id -> {'T': world pose, 'num_cams', 'reproj_px', ...}
               (output of ``tag_world_poses``).
    ``rig``:   output of ``build_tag_rig`` (or the loaded JSON form).

    Each visible modelled tag votes: T_body_world_vote = T_world_tag @ inv(T_body_tag).
    Votes are robustly fused: translation median + inlier (<= max_trans_dev_m)
    reject, then weighted quaternion average of the inliers.

    Returns None if no modelled tag is visible, else:
        {'R': 3x3, 't': 3, 'num_tags': int, 'tags': [ids], 'reproj_px': float}
    """
    rig_tags = rig['tags']
    votes = []       # (T_world_body, weight, tag_id, reproj)
    for tid, info in poses.items():
        if tid not in rig_tags:
            continue
        T_body_tag = np.asarray(rig_tags[tid]['T_body_tag'], dtype=np.float64)
        T_world_body = info['T'] @ invert_pose(T_body_tag)
        weight = float(info.get('num_cams', 1)) / (info.get('reproj_px', 1.0) + 0.5)
        votes.append((T_world_body, weight, tid, info.get('reproj_px', 0.0)))

    if not votes:
        return None

    trans = np.array([v[0][:3, 3] for v in votes])
    if len(votes) >= 3:
        med = np.median(trans, axis=0)
        keep = np.linalg.norm(trans - med, axis=1) <= max_trans_dev_m
        if keep.sum() >= 1:
            votes = [v for v, k in zip(votes, keep) if k]

    poses_in = [v[0] for v in votes]
    weights = [v[1] for v in votes]
    tags_used = [v[2] for v in votes]
    reproj = max(v[3] for v in votes)
    T = average_poses(poses_in, weights)
    return {
        'R': T[:3, :3],
        't': T[:3, 3],
        'num_tags': len(votes),
        'tags': tags_used,
        'reproj_px': float(reproj),
    }
