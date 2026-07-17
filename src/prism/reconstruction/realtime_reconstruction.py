import os
import sys

import numpy as np

from prism.common.timebase import pick_bracket, pick_nearest


DEFAULT_LEGACY_RECORDING_DIR = '/opt/MVS/Samples/64/Python/General/Recording'

COLOR_ORDER = ['yellow', 'blue', 'green']
COLOR_BRG = {
    'yellow': (0, 220, 255),
    'blue': (255, 120, 0),
    'green': (0, 220, 0),
}
COLOR_MPL = {
    'yellow': 'gold',
    'blue': 'tab:blue',
    'green': 'tab:green',
}


def _ensure_legacy_recording_path():
    legacy_dir = os.environ.get('PRISM_LEGACY_RECORDING_DIR', DEFAULT_LEGACY_RECORDING_DIR)
    if legacy_dir and legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)


def _legacy_led_module():
    _ensure_legacy_recording_path()
    import LedRigidBody6D4CamHSV3Color as legacy_led
    return legacy_led


def detect_all_colors(frame, hsv_cfg, min_area, cache):
    key = id(frame)
    if key in cache:
        return cache[key]

    legacy_led = _legacy_led_module()
    res = {}
    for name in COLOR_ORDER:
        low, high = hsv_cfg[name]
        pt, _ = legacy_led.detect_led_hsv(frame, low, high, min_area)
        res[name] = pt
    cache[key] = res
    return res


def make_track_state():
    legacy_led = _legacy_led_module()
    return {
        'kalman': {name: legacy_led.create_kalman() for name in COLOR_ORDER},
        'state_inited': {name: False for name in COLOR_ORDER},
        'pred_miss_count': {name: 0 for name in COLOR_ORDER},
        'last_point_valid': {name: False for name in COLOR_ORDER},
        'traj_points': {name: [] for name in COLOR_ORDER},
    }


def advance_tracking(st, observations_by_color, cameras, args, dt, now, t0, writer, paused):
    legacy_led = _legacy_led_module()
    kalman = st['kalman']
    state_inited = st['state_inited']
    pred_miss_count = st['pred_miss_count']
    last_point_valid = st['last_point_valid']
    traj_points = st['traj_points']

    preds = {}
    for name in COLOR_ORDER:
        legacy_led.update_transition_dt(kalman[name], dt)
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
                x_meas, err_dict = legacy_led.robust_triangulate(observations, cameras, args.max_norm_reproj_error)

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
                    kalman[name] = legacy_led.create_kalman()
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
