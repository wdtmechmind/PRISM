import cv2
import numpy as np

from prism.common.timebase import pick_bracket, pick_nearest


COLOR_ORDER = ['red', 'yellow', 'blue', 'green']
COLOR_BRG = {
    'red': (0, 0, 255),
    'yellow': (0, 220, 255),
    'blue': (255, 120, 0),
    'green': (0, 220, 0),
}
COLOR_MPL = {
    'red': 'tab:red',
    'yellow': 'gold',
    'blue': 'tab:blue',
    'green': 'tab:green',
}


def normalize_vec(vec):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vec / norm


def matrix_to_rpy_zyx(rot):
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    sy = -float(rot[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = float(np.arcsin(sy))
    cp = float(np.cos(pitch))

    if abs(cp) > 1e-8:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        roll = 0.0
        yaw = float(np.arctan2(-rot[0, 1], rot[1, 1]))

    return roll, pitch, yaw


def _select_body_frame_colors(world_points_by_color):
    preferred = [name for name in ['yellow', 'blue', 'green'] if name in world_points_by_color]
    if len(preferred) >= 3:
        return preferred[:3]

    available = [name for name in COLOR_ORDER if name in world_points_by_color]
    if len(available) < 3:
        return []
    return available[:3]


def build_body_model(world_points_by_color):
    frame_colors = _select_body_frame_colors(world_points_by_color)
    if len(frame_colors) < 3:
        return None

    p0 = np.asarray(world_points_by_color[frame_colors[0]], dtype=np.float64).reshape(3)
    p1 = np.asarray(world_points_by_color[frame_colors[1]], dtype=np.float64).reshape(3)
    p2 = np.asarray(world_points_by_color[frame_colors[2]], dtype=np.float64).reshape(3)

    model_colors = [name for name in COLOR_ORDER if name in world_points_by_color]
    origin = np.mean([np.asarray(world_points_by_color[name], dtype=np.float64).reshape(3) for name in model_colors], axis=0)
    ex = normalize_vec(p1 - p0)
    vg = p2 - p0
    ez_raw = np.cross(ex, vg)
    if np.linalg.norm(ez_raw) < 1e-10:
        return None
    ez = normalize_vec(ez_raw)
    ey = normalize_vec(np.cross(ez, ex))

    rot_wb = np.column_stack([ex, ey, ez])
    model = {}
    for name in model_colors:
        pw = np.asarray(world_points_by_color[name], dtype=np.float64).reshape(3)
        model[name] = rot_wb.T @ (pw - origin)

    return {
        'model_points': model,
        'init_origin': origin,
        'init_rot_wb': rot_wb,
        'frame_colors': frame_colors,
    }


def update_body_model(model_points_by_color, observed_points_by_color, pose_rot, pose_trans):
    pose_rot = np.asarray(pose_rot, dtype=np.float64).reshape(3, 3)
    pose_trans = np.asarray(pose_trans, dtype=np.float64).reshape(3)
    added = []
    for name in COLOR_ORDER:
        if name in model_points_by_color or name not in observed_points_by_color:
            continue
        pw = np.asarray(observed_points_by_color[name], dtype=np.float64).reshape(3)
        model_points_by_color[name] = pose_rot.T @ (pw - pose_trans)
        added.append(name)
    return added


def estimate_pose_from_model(model_points_by_color, observed_points_by_color):
    names = [name for name in COLOR_ORDER if name in model_points_by_color and name in observed_points_by_color]
    if len(names) < 3:
        return None

    model_points = np.asarray([model_points_by_color[name] for name in names], dtype=np.float64)
    observed_points = np.asarray([observed_points_by_color[name] for name in names], dtype=np.float64)

    model_center = np.mean(model_points, axis=0)
    observed_center = np.mean(observed_points, axis=0)
    model_zero = model_points - model_center
    observed_zero = observed_points - observed_center

    H = model_zero.T @ observed_zero
    U, _, Vt = np.linalg.svd(H)
    rot = Vt.T @ U.T
    if np.linalg.det(rot) < 0.0:
        Vt[-1, :] *= -1.0
        rot = Vt.T @ U.T

    trans = observed_center - rot @ model_center
    return rot, trans


def detect_led_hsv(img_bgr, hsv_low, hsv_high, min_area):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_low, dtype=np.uint8)
    upper = np.array(hsv_high, dtype=np.uint8)
    if int(lower[0]) <= int(upper[0]):
        mask = cv2.inRange(hsv, lower, upper)
    else:
        low_a = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
        high_a = np.array([179, upper[1], upper[2]], dtype=np.uint8)
        low_b = np.array([0, lower[1], lower[2]], dtype=np.uint8)
        high_b = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
        mask = cv2.bitwise_or(cv2.inRange(hsv, low_a, high_a), cv2.inRange(hsv, low_b, high_b))

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    best = None
    best_area = 0.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        if area > best_area:
            best = c
            best_area = area

    if best is None:
        return None, 0.0

    m = cv2.moments(best)
    if abs(m['m00']) < 1e-9:
        return None, 0.0

    cx = float(m['m10'] / m['m00'])
    cy = float(m['m01'] / m['m00'])
    return (cx, cy), best_area


def create_kalman():
    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.eye(6, dtype=np.float32)
    kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
    kf.measurementMatrix[0, 0] = 1.0
    kf.measurementMatrix[1, 1] = 1.0
    kf.measurementMatrix[2, 2] = 1.0
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-4
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


def update_transition_dt(kf, dt):
    dt = max(1e-3, float(dt))
    f = np.eye(6, dtype=np.float32)
    f[0, 3] = dt
    f[1, 4] = dt
    f[2, 5] = dt
    kf.transitionMatrix = f


def triangulate_multi_view(observations, cameras):
    if len(observations) < 2:
        return None, None

    A_rows = []
    used = []
    und_points = {}

    for cam_idx, (u, v) in observations.items():
        cam = cameras[cam_idx]
        uv = np.array([[[u, v]]], dtype=np.float64)
        und = cv2.undistortPoints(uv, cam['K'], cam['D']).reshape(2)
        und_points[cam_idx] = und

        p = np.hstack([cam['R'], cam['t'].reshape(3, 1)])
        A_rows.append(und[0] * p[2, :] - p[0, :])
        A_rows.append(und[1] * p[2, :] - p[1, :])
        used.append(cam_idx)

    A = np.asarray(A_rows, dtype=np.float64)
    if A.shape[0] < 4:
        return None, None

    _, _, vt = np.linalg.svd(A)
    x_h = vt[-1, :]
    if abs(x_h[3]) < 1e-12:
        return None, None

    x = x_h[:3] / x_h[3]
    errs = {}
    for cam_idx in used:
        cam = cameras[cam_idx]
        xc = cam['R'] @ x.reshape(3, 1) + cam['t'].reshape(3, 1)
        z = float(xc[2, 0])
        if z <= 1e-9:
            return None, None
        pred = np.array([xc[0, 0] / z, xc[1, 0] / z], dtype=np.float64)
        errs[cam_idx] = float(np.linalg.norm(pred - und_points[cam_idx]))

    return x, errs


def robust_triangulate(observations, cameras, max_norm_reproj_error):
    if len(observations) < 2:
        return None, None

    x, errs = triangulate_multi_view(observations, cameras)
    if x is None:
        return None, None

    max_err = max(errs.values()) if errs else 1e9
    if max_err <= max_norm_reproj_error or len(observations) == 2:
        return x, errs

    best = (x, errs)
    best_max = max_err
    keys = sorted(observations.keys())
    for drop in keys:
        sub_obs = {k: v for k, v in observations.items() if k != drop}
        if len(sub_obs) < 2:
            continue
        x_sub, errs_sub = triangulate_multi_view(sub_obs, cameras)
        if x_sub is None or not errs_sub:
            continue
        m = max(errs_sub.values())
        if m < best_max:
            best = (x_sub, errs_sub)
            best_max = m

    if best_max > max_norm_reproj_error:
        return None, None
    return best


def detect_all_colors(frame, hsv_cfg, min_area, cache):
    key = id(frame)
    if key in cache:
        return cache[key]

    res = {}
    for name in COLOR_ORDER:
        low, high = hsv_cfg[name]
        pt, _ = detect_led_hsv(frame, low, high, min_area)
        res[name] = pt
    cache[key] = res
    return res


def make_track_state():
    return {
        'kalman': {name: create_kalman() for name in COLOR_ORDER},
        'state_inited': {name: False for name in COLOR_ORDER},
        'pred_miss_count': {name: 0 for name in COLOR_ORDER},
        'last_point_valid': {name: False for name in COLOR_ORDER},
        'traj_points': {name: [] for name in COLOR_ORDER},
    }


def advance_tracking(st, observations_by_color, cameras, args, dt, now, t0, writer, paused):
    kalman = st['kalman']
    state_inited = st['state_inited']
    pred_miss_count = st['pred_miss_count']
    last_point_valid = st['last_point_valid']
    traj_points = st['traj_points']

    preds = {}
    for name in COLOR_ORDER:
        update_transition_dt(kalman[name], dt)
        preds[name] = kalman[name].predict().reshape(-1)

    mode_by_color = {name: 'none' for name in COLOR_ORDER}
    point_by_color = {name: None for name in COLOR_ORDER}
    max_err_by_color = {name: '' for name in COLOR_ORDER}

    if paused:
        for name in COLOR_ORDER:
            mode_by_color[name] = 'paused'
    else:
        for name in COLOR_ORDER:
            observations = observations_by_color[name]
            x_meas = None
            err_dict = None
            if len(observations) >= 2:
                x_meas, err_dict = robust_triangulate(observations, cameras, args.max_norm_reproj_error)

            if x_meas is not None:
                z = np.asarray(x_meas, dtype=np.float32).reshape(3, 1)
                if not state_inited[name]:
                    kalman[name].statePost = np.array(
                        [[z[0, 0]], [z[1, 0]], [z[2, 0]], [0.0], [0.0], [0.0]],
                        dtype=np.float32,
                    )
                    state_inited[name] = True
                corrected = kalman[name].correct(z).reshape(-1)
                point_by_color[name] = corrected[:3].astype(np.float64)
                mode_by_color[name] = 'measured'
                pred_miss_count[name] = 0
                if err_dict:
                    max_err_by_color[name] = '%.6f' % max(err_dict.values())
            elif state_inited[name]:
                pred_miss_count[name] += 1
                if pred_miss_count[name] <= int(args.max_predict_frames):
                    point_by_color[name] = preds[name][:3].astype(np.float64)
                    mode_by_color[name] = 'predicted'
                else:
                    mode_by_color[name] = 'lost'
                    point_by_color[name] = None
                    state_inited[name] = False
                    kalman[name] = create_kalman()
                    pred_miss_count[name] = 0

    for name in COLOR_ORDER:
        point_out = point_by_color[name]
        observations = observations_by_color[name]
        mode = mode_by_color[name]
        max_err = max_err_by_color[name]
        if point_out is not None:
            traj_points[name].append(point_out.tolist())
            last_point_valid[name] = True
            if len(traj_points[name]) > args.max_traj_points:
                traj_points[name] = traj_points[name][-args.max_traj_points:]
            vis = ','.join(['cam%d' % i for i in sorted(observations.keys())])
            writer.writerow([
                '%.6f' % (now - t0), name,
                '%.9f' % point_out[0], '%.9f' % point_out[1], '%.9f' % point_out[2],
                mode, len(observations), max_err, vis,
            ])
        elif last_point_valid[name] and mode in ('lost', 'paused'):
            traj_points[name].append([float('nan'), float('nan'), float('nan')])
            if len(traj_points[name]) > args.max_traj_points:
                traj_points[name] = traj_points[name][-args.max_traj_points:]
            last_point_valid[name] = False

    return point_by_color, mode_by_color


def build_observations_nearest(buffers, t_ref, hsv_cfg, min_area, det_cache):
    obs = {name: {} for name in COLOR_ORDER}
    for cam_i in range(4):
        buf = buffers[cam_i]
        if not buf:
            continue
        sel = pick_nearest(buf, t_ref)
        if sel is None:
            continue
        det = detect_all_colors(sel[1], hsv_cfg, min_area, det_cache)
        for name in COLOR_ORDER:
            if det[name] is not None:
                obs[name][cam_i] = det[name]
    return obs


def build_observations_interp(buffers, t_ref, hsv_cfg, min_area, det_cache):
    obs = {name: {} for name in COLOR_ORDER}
    for cam_i in range(4):
        buf = buffers[cam_i]
        if not buf:
            continue
        before, after = pick_bracket(buf, t_ref)
        if before is not None and after is not None and after[0] > before[0]:
            det_b = detect_all_colors(before[1], hsv_cfg, min_area, det_cache)
            det_a = detect_all_colors(after[1], hsv_cfg, min_area, det_cache)
            w = (t_ref - before[0]) / (after[0] - before[0])
            for name in COLOR_ORDER:
                pb = det_b[name]
                pa = det_a[name]
                if pb is not None and pa is not None:
                    obs[name][cam_i] = (pb[0] + w * (pa[0] - pb[0]), pb[1] + w * (pa[1] - pb[1]))
                elif pb is not None:
                    obs[name][cam_i] = pb
                elif pa is not None:
                    obs[name][cam_i] = pa
        else:
            one = before if before is not None else after
            if one is None:
                continue
            det = detect_all_colors(one[1], hsv_cfg, min_area, det_cache)
            for name in COLOR_ORDER:
                if det[name] is not None:
                    obs[name][cam_i] = det[name]
    return obs
