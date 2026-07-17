import cv2
import numpy as np

from prism.reconstruction.realtime_reconstruction import COLOR_BRG, COLOR_ORDER


def draw_preview(hik_latest, rs_latest, observations_by_color, hik_fps, rs_fps,
                 recording, trial_id, elapsed_sec, target_w=480):
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
    return grid
