import cv2
import numpy as np

from prism.reconstruction.realtime_reconstruction import COLOR_BRG, COLOR_ORDER


def _pad_to_height(img, target_h):
    if img.shape[0] == target_h:
        return img
    if img.shape[0] > target_h:
        return img[:target_h, :, :]
    pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return np.vstack([img, pad])


def _build_hand_panel(width, hand_info=None):
    panel_h = 180
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, panel_h - 1), (70, 70, 70), 1)
    cv2.putText(panel, 'Hand Control', (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    if hand_info is None:
        hand_info = {}

    connected = bool(hand_info.get('connected', False))
    status = 'CONNECTED' if connected else 'DISCONNECTED'
    status_color = (80, 220, 80) if connected else (70, 170, 255)
    cv2.putText(panel, 'socket: %s' % status, (14, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, status_color, 2, cv2.LINE_AA)

    last_action = hand_info.get('last_action', 'none')
    last_cmd = hand_info.get('last_command', '')
    last_t = hand_info.get('last_t_sec', float('nan'))
    total_cmd = int(hand_info.get('total_commands', 0))
    cv2.putText(panel, 'last action: %s' % last_action, (14, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(panel, 'last cmd: %s' % (last_cmd if last_cmd else 'n/a'), (14, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    if last_t == last_t:
        t_text = 'last t_sec: %.6f' % float(last_t)
    else:
        t_text = 'last t_sec: n/a'
    cv2.putText(panel, t_text, (14, 144),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, 'commands sent: %d' % total_cmd, (14, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def draw_preview(hik_latest, rs_latest, observations_by_color, hik_fps, rs_fps,
                 recording, trial_id, elapsed_sec, target_w=480,
                 traj_image=None, hand_info=None, traj_error=''):
    cells = []
    for i in range(4):
        img = hik_latest[i]
        if img is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        c = img.copy()
        for name in COLOR_ORDER:
            obs = observations_by_color[name]
            if i in obs:
                u, v = obs[i]
                cv2.circle(c, (int(round(u)), int(round(v))), 8, COLOR_BRG[name], 2)
        cv2.putText(c, 'hik%d %.1f fps' % (i, hik_fps[i]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        h, w = c.shape[:2]
        s = target_w / float(w)
        cells.append(cv2.resize(c, (target_w, max(1, int(h * s))), interpolation=cv2.INTER_AREA))

    rs_show = rs_latest.copy() if rs_latest is not None else np.zeros_like(cells[0])
    cv2.putText(rs_show, 'realsense %.1f fps' % rs_fps, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 255), 2, cv2.LINE_AA)
    hh, ww = rs_show.shape[:2]
    s = target_w / float(ww)
    rs_cell = cv2.resize(rs_show, (target_w, max(1, int(hh * s))), interpolation=cv2.INTER_AREA)

    max_h = max([c.shape[0] for c in cells] + [rs_cell.shape[0]])

    def pad(c):
        if c.shape[0] < max_h:
            p = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
            return np.vstack([c, p])
        return c

    cells = [pad(c) for c in cells]
    rs_cell = pad(rs_cell)

    top = np.hstack([cells[0], cells[1]])
    mid = np.hstack([cells[2], cells[3]])
    bottom = np.hstack([rs_cell, np.zeros_like(rs_cell)])
    grid = np.vstack([top, mid, bottom])

    rec_txt = 'REC trial_%06d %.1fs' % (trial_id, elapsed_sec) if recording else 'IDLE (SPACE to start trial)'
    rec_color = (0, 0, 255) if recording else (200, 200, 200)
    cv2.putText(grid, rec_txt, (10, grid.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, rec_color, 2, cv2.LINE_AA)

    if traj_image is None:
        side_w = max(560, int(grid.shape[1] * 0.8))
        side = np.zeros((grid.shape[0], side_w, 3), dtype=np.uint8)
        msg_y = 220
        cv2.putText(side, '3D trajectory preview unavailable', (20, msg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        if traj_error:
            short_err = traj_error[:88]
            cv2.putText(side, 'reason: %s' % short_err, (20, msg_y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 255), 1, cv2.LINE_AA)
    else:
        side_w = max(560, int(grid.shape[1] * 0.8))
        traj_h, traj_w = traj_image.shape[:2]
        scale = min(side_w / float(traj_w), (grid.shape[0] - 180) / float(max(1, traj_h)))
        show_w = max(1, int(traj_w * scale))
        show_h = max(1, int(traj_h * scale))
        traj_show = cv2.resize(traj_image, (show_w, show_h), interpolation=cv2.INTER_AREA)
        side = np.zeros((grid.shape[0], side_w, 3), dtype=np.uint8)
        x0 = max(0, (side_w - show_w) // 2)
        y0 = 180
        y1 = min(side.shape[0], y0 + show_h)
        x1 = min(side.shape[1], x0 + show_w)
        side[y0:y1, x0:x1, :] = traj_show[:(y1 - y0), :(x1 - x0), :]

    hand_panel = _build_hand_panel(side.shape[1], hand_info=hand_info)
    side[:hand_panel.shape[0], :, :] = hand_panel

    grid = _pad_to_height(grid, max(grid.shape[0], side.shape[0]))
    side = _pad_to_height(side, grid.shape[0])
    return np.hstack([grid, side])
