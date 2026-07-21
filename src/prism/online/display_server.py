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


def _resize_with_quality(img, out_w, out_h):
    in_h, in_w = img.shape[:2]
    if out_w <= 0 or out_h <= 0:
        return img
    if out_w == in_w and out_h == in_h:
        return img

    # Prefer area for shrinking, cubic for enlarging for clearer details.
    if out_w < in_w or out_h < in_h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(img, (out_w, out_h), interpolation=interp)


def _draw_gradient_background(img, top_bgr, bottom_bgr):
    h, w = img.shape[:2]
    for y in range(h):
        t = float(y) / float(max(1, h - 1))
        b = int(round((1.0 - t) * top_bgr[0] + t * bottom_bgr[0]))
        g = int(round((1.0 - t) * top_bgr[1] + t * bottom_bgr[1]))
        r = int(round((1.0 - t) * top_bgr[2] + t * bottom_bgr[2]))
        img[y, :, :] = (b, g, r)


def _parse_last_gesture_id(last_action):
    if not last_action or ':' not in last_action:
        return None
    head = last_action.split(':', 1)[0].strip()
    if not head.isdigit():
        return None
    return int(head)


def _build_hand_panel(width, hand_info=None):
    panel_h = 620
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    _draw_gradient_background(panel, (30, 34, 40), (20, 22, 27))
    cv2.rectangle(panel, (0, 0), (width - 1, panel_h - 1), (75, 85, 100), 1)

    title_h = 46
    cv2.rectangle(panel, (0, 0), (width - 1, title_h), (44, 52, 66), -1)
    cv2.putText(panel, 'DexHand Control Console', (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)

    if hand_info is None:
        hand_info = {}

    connected = bool(hand_info.get('connected', False))
    status = 'CONNECTED' if connected else 'DISCONNECTED'
    status_bg = (65, 150, 65) if connected else (65, 92, 150)
    cv2.rectangle(panel, (14, 58), (220, 92), status_bg, -1)
    cv2.rectangle(panel, (14, 58), (220, 92), (120, 130, 150), 1)
    cv2.putText(panel, 'SOCKET: %s' % status, (24, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 2, cv2.LINE_AA)

    last_action = hand_info.get('last_action', 'none')
    last_cmd = hand_info.get('last_command', '')
    last_t = hand_info.get('last_t_sec', float('nan'))
    total_cmd = int(hand_info.get('total_commands', 0))

    cv2.putText(panel, 'last action: %s' % last_action, (244, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(panel, 'last cmd: %s' % (last_cmd if last_cmd else 'n/a'), (244, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (205, 205, 205), 1, cv2.LINE_AA)
    if last_t == last_t:
        t_text = 'last t_sec: %.6f' % float(last_t)
    else:
        t_text = 'last t_sec: n/a'
    cv2.putText(panel, t_text, (14, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, 'commands sent: %d' % total_cmd, (14, 146),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (205, 205, 205), 1, cv2.LINE_AA)

    pending_digits = str(hand_info.get('pending_digits', '') or '-')
    cv2.putText(panel, 'pending digits: %s' % pending_digits, (14, 172),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Click a button or type 1-17 in preview', (14, 194),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 190, 205), 1, cv2.LINE_AA)
    cv2.putText(panel, 'Enter=submit pending 1 | Backspace=clear', (14, 214),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 190, 205), 1, cv2.LINE_AA)

    cv2.putText(panel, 'Gesture Buttons (1..17)', (14, 248),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)

    hints = hand_info.get('gesture_hints', [])
    margin = 14
    gap_x = 10
    gap_y = 8
    cols = 3
    button_w = max(170, int((width - margin * 2 - gap_x * (cols - 1)) / cols))
    button_h = 42
    row_y_start = 272
    button_regions = []
    last_gid = _parse_last_gesture_id(last_action)

    for idx, item in enumerate(hints):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        gid = int(item[0])
        name = str(item[1])
        col = idx % cols
        row = idx // cols
        x = margin + col * (button_w + gap_x)
        y = row_y_start + row * (button_h + gap_y)
        x2 = min(width - margin, x + button_w)
        y2 = y + button_h

        is_last = (last_gid == gid)
        fill = (72, 84, 104) if not is_last else (86, 112, 76)
        edge = (118, 136, 164) if not is_last else (160, 206, 118)
        cv2.rectangle(panel, (x, y), (x2, y2), fill, -1)
        cv2.rectangle(panel, (x, y), (x2, y2), edge, 1)

        cv2.putText(panel, '%02d' % gid, (x + 10, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        short_name = name if len(name) <= 19 else (name[:16] + '...')
        cv2.putText(panel, short_name, (x + 56, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (236, 236, 236), 1, cv2.LINE_AA)

        button_regions.append({
            'gesture_id': gid,
            'x1': x,
            'y1': y,
            'x2': x2,
            'y2': y2,
        })

    return panel, button_regions


def draw_preview(hik_latest, rs_latest, observations_by_color, hik_fps, rs_fps,
                 recording, trial_id, elapsed_sec, target_w=480,
                 traj_image=None, hand_info=None, traj_error='', return_ui_meta=False,
                 composite_target_w=0):
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
        cells.append(_resize_with_quality(c, target_w, max(1, int(h * s))))

    rs_show = rs_latest.copy() if rs_latest is not None else np.zeros_like(cells[0])
    cv2.putText(rs_show, 'realsense %.1f fps' % rs_fps, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 255, 255), 2, cv2.LINE_AA)
    hh, ww = rs_show.shape[:2]
    s = target_w / float(ww)
    rs_cell = _resize_with_quality(rs_show, target_w, max(1, int(hh * s)))

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

    if composite_target_w and composite_target_w > grid.shape[1]:
        # Keep final composed width close to window width to avoid secondary downscaling by imshow.
        side_w = max(480, int(composite_target_w - grid.shape[1]))
    else:
        side_w = max(640, int(grid.shape[1] * 0.78))
    hand_panel, button_regions = _build_hand_panel(side_w, hand_info=hand_info)
    hand_panel_h = hand_panel.shape[0]

    if traj_image is None:
        side = np.zeros((grid.shape[0], side_w, 3), dtype=np.uint8)
        msg_y = min(grid.shape[0] - 80, hand_panel_h + 40)
        cv2.putText(side, '3D trajectory preview unavailable', (20, msg_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        if traj_error:
            short_err = traj_error[:88]
            cv2.putText(side, 'reason: %s' % short_err, (20, msg_y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 255), 1, cv2.LINE_AA)
    else:
        traj_h, traj_w = traj_image.shape[:2]
        scale = min(side_w / float(traj_w), (grid.shape[0] - hand_panel_h) / float(max(1, traj_h)))
        show_w = max(1, int(traj_w * scale))
        show_h = max(1, int(traj_h * scale))
        traj_show = _resize_with_quality(traj_image, show_w, show_h)
        side = np.zeros((grid.shape[0], side_w, 3), dtype=np.uint8)
        x0 = max(0, (side_w - show_w) // 2)
        y0 = hand_panel_h
        y1 = min(side.shape[0], y0 + show_h)
        x1 = min(side.shape[1], x0 + show_w)
        side[y0:y1, x0:x1, :] = traj_show[:(y1 - y0), :(x1 - x0), :]

    side[:hand_panel.shape[0], :, :] = hand_panel

    grid = _pad_to_height(grid, max(grid.shape[0], side.shape[0]))
    side = _pad_to_height(side, grid.shape[0])

    out = np.hstack([grid, side])
    if not return_ui_meta:
        return out

    x_offset = grid.shape[1]
    regions_out = []
    for region in button_regions:
        regions_out.append({
            'gesture_id': region['gesture_id'],
            'x1': x_offset + int(region['x1']),
            'y1': int(region['y1']),
            'x2': x_offset + int(region['x2']),
            'y2': int(region['y2']),
        })
    return out, {'gesture_regions': regions_out}
