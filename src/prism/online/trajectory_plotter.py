import numpy as np

from prism.reconstruction.calibration import apply_corrected_transform
from prism.reconstruction.realtime_reconstruction import COLOR_MPL, COLOR_ORDER


class Live3DPlotter(object):
    def __init__(self, enabled=True, camera_centers=None, corrected_transform=None):
        self.enabled = bool(enabled)
        self.ready = False
        self.fig = None
        self.ax_raw = None
        self.ax_corr = None
        self.camera_centers = camera_centers if camera_centers is not None else {}
        self.corrected_transform = corrected_transform
        self.cam_scatter_raw = None
        self.cam_scatter_corr = None
        self.cam_labels_raw = []
        self.cam_labels_corr = []
        self.lines_raw = {}
        self.lines_corr = {}
        self.current_raw = {}
        self.current_corr = {}
        self.rigid_center_line_raw = None
        self.rigid_center_line_corr = None
        self.rigid_origin_raw = None
        self.rigid_origin_corr = None
        self.rigid_axes_raw = {}
        self.rigid_axes_corr = {}
        self.rigid_axes_hist_raw = {}
        self.rigid_axes_hist_corr = {}
        self.latest_bgr = None
        self.latest_error = ''

        if not self.enabled:
            return

        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            self.Figure = Figure
            self.FigureCanvasAgg = FigureCanvasAgg
            self.Line3DCollection = Line3DCollection
        except Exception as ex:
            print('warning: matplotlib unavailable, disable 3D plot: %s' % ex)
            self.enabled = False
            return

        self.fig = self.Figure(figsize=(8, 10), dpi=120)
        self.canvas = self.FigureCanvasAgg(self.fig)
        self.ax_raw = self.fig.add_subplot(211, projection='3d')
        self.ax_raw.set_xlabel('X (m)')
        self.ax_raw.set_ylabel('Y (m)')
        self.ax_raw.set_zlabel('Z (m)')
        self.ax_raw.set_title('Raw World Frame')

        self.ax_corr = self.fig.add_subplot(212, projection='3d')
        self.ax_corr.set_xlabel('Xc (m)')
        self.ax_corr.set_ylabel('Yc (m)')
        self.ax_corr.set_zlabel('Zc (m)')
        self.ax_corr.set_title('Corrected Frame (camera plane z=0)')

        for ax in [self.ax_raw, self.ax_corr]:
            ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.45)
            ax.view_init(elev=22, azim=-58)
            ax.tick_params(labelsize=8)

        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.04, hspace=0.22)

        for name in COLOR_ORDER:
            c = COLOR_MPL[name]
            self.lines_raw[name], = self.ax_raw.plot([], [], [], '-', color=c, linewidth=2.4, label=name)
            self.current_raw[name] = self.ax_raw.scatter([], [], [], color=c, s=48)
            self.lines_corr[name], = self.ax_corr.plot([], [], [], '-', color=c, linewidth=2.4, label=name)
            self.current_corr[name] = self.ax_corr.scatter([], [], [], color=c, s=48)

        self.rigid_center_line_raw, = self.ax_raw.plot([], [], [], '--', color='magenta', linewidth=2.0, label='rigid-center')
        self.rigid_center_line_corr, = self.ax_corr.plot([], [], [], '--', color='magenta', linewidth=2.0, label='rigid-center')
        self.rigid_origin_raw = self.ax_raw.scatter([], [], [], color='black', s=52, marker='o')
        self.rigid_origin_corr = self.ax_corr.scatter([], [], [], color='black', s=52, marker='o')

        self.rigid_axes_raw['x'], = self.ax_raw.plot([], [], [], '-', color='red', linewidth=2.8)
        self.rigid_axes_raw['y'], = self.ax_raw.plot([], [], [], '-', color='green', linewidth=2.8)
        self.rigid_axes_raw['z'], = self.ax_raw.plot([], [], [], '-', color='blue', linewidth=2.8)
        self.rigid_axes_corr['x'], = self.ax_corr.plot([], [], [], '-', color='red', linewidth=2.8)
        self.rigid_axes_corr['y'], = self.ax_corr.plot([], [], [], '-', color='green', linewidth=2.8)
        self.rigid_axes_corr['z'], = self.ax_corr.plot([], [], [], '-', color='blue', linewidth=2.8)

        hist_style = {
            'x': ('red', 0.9),
            'y': ('green', 0.9),
            'z': ('blue', 0.9),
        }
        dummy_seg = [np.zeros((2, 3), dtype=np.float64)]
        for axis in ['x', 'y', 'z']:
            color, alpha = hist_style[axis]
            self.rigid_axes_hist_raw[axis] = self.Line3DCollection(dummy_seg, colors=color, linewidths=1.0, alpha=alpha * 0.35)
            self.rigid_axes_hist_corr[axis] = self.Line3DCollection(dummy_seg, colors=color, linewidths=1.0, alpha=alpha * 0.35)
            self.ax_raw.add_collection3d(self.rigid_axes_hist_raw[axis])
            self.ax_corr.add_collection3d(self.rigid_axes_hist_corr[axis])
            self.rigid_axes_hist_raw[axis].set_segments([])
            self.rigid_axes_hist_corr[axis].set_segments([])

        self.ax_raw.legend(loc='upper right', fontsize=8, framealpha=0.8)
        self.ax_corr.legend(loc='upper right', fontsize=8, framealpha=0.8)

        self._draw_camera_markers()
        self.ready = True

    def _canvas_to_bgr(self):
        # Prefer RGBA buffer path (stable across newer Matplotlib versions).
        rgba = np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)
        if rgba.ndim == 3 and rgba.shape[2] == 4:
            rgb = rgba[:, :, :3]
            return rgb[:, :, ::-1].copy()

        # Fallback for older Matplotlib APIs.
        w, h = self.fig.canvas.get_width_height()
        rgb_buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        rgb = rgb_buf.reshape(h, w, 3)
        return rgb[:, :, ::-1].copy()

    @staticmethod
    def _set_line3d(line, p0, p1):
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    def _hide_rigid_axes(self):
        if self.rigid_origin_raw is not None:
            self.rigid_origin_raw._offsets3d = ([], [], [])
        if self.rigid_origin_corr is not None:
            self.rigid_origin_corr._offsets3d = ([], [], [])
        for axis in ['x', 'y', 'z']:
            self._set_line3d(self.rigid_axes_raw[axis], np.zeros(3), np.zeros(3))
            self._set_line3d(self.rigid_axes_corr[axis], np.zeros(3), np.zeros(3))

    @staticmethod
    def _build_axis_segments(pose_history, pose_rot_history, axis_len):
        segs = {'x': [], 'y': [], 'z': []}
        if pose_history is None or pose_rot_history is None:
            return segs

        n = min(len(pose_history), len(pose_rot_history))
        if n <= 0:
            return segs

        axes = [('x', 0), ('y', 1), ('z', 2)]
        for i in range(n):
            p = np.asarray(pose_history[i], dtype=np.float64).reshape(-1)
            r = np.asarray(pose_rot_history[i], dtype=np.float64).reshape(-1)
            if p.size < 3 or r.size < 9:
                continue
            p0 = p[:3]
            if not np.isfinite(p0).all():
                continue
            R = r[:9].reshape(3, 3)
            if not np.isfinite(R).all():
                continue

            for axis_name, axis_idx in axes:
                p1 = p0 + axis_len * R[:, axis_idx]
                segs[axis_name].append(np.vstack([p0, p1]))

        return segs

    def _draw_camera_markers(self):
        if self.ax_raw is None:
            return
        if len(self.camera_centers) == 0:
            return

        cam_ids = sorted(self.camera_centers.keys())
        arr = np.asarray([self.camera_centers[i] for i in cam_ids], dtype=np.float64)
        self.cam_scatter_raw = self.ax_raw.scatter(arr[:, 0], arr[:, 1], arr[:, 2], color='black', s=30, marker='^')
        for idx, cid in enumerate(cam_ids):
            x, y, z = arr[idx]
            lbl = self.ax_raw.text(x, y, z, 'cam%d' % cid, color='black', fontsize=8)
            self.cam_labels_raw.append(lbl)

        if self.corrected_transform is not None and self.ax_corr is not None:
            carr = apply_corrected_transform(arr, self.corrected_transform)
            self.cam_scatter_corr = self.ax_corr.scatter(carr[:, 0], carr[:, 1], carr[:, 2], color='black', s=30, marker='^')
            for idx, cid in enumerate(cam_ids):
                x, y, z = carr[idx]
                lbl = self.ax_corr.text(x, y, z, 'cam%d' % cid, color='black', fontsize=8)
                self.cam_labels_corr.append(lbl)

    def _set_equal_limits(self, ax, pts_list, camera_pts=None):
        fpt_blocks = []
        for pts in pts_list:
            p = np.asarray(pts, dtype=np.float64)
            if p.size == 0:
                continue
            finite = np.isfinite(p).all(axis=1)
            if np.any(finite):
                fpt_blocks.append(p[finite])

        if camera_pts is not None and len(camera_pts) > 0:
            fpt_blocks.append(np.asarray(camera_pts, dtype=np.float64))

        if not fpt_blocks:
            return

        fpts = np.vstack(fpt_blocks)
        mins = np.min(fpts, axis=0)
        maxs = np.max(fpts, axis=0)
        center = 0.5 * (mins + maxs)
        span = np.max(np.maximum(maxs - mins, 1e-6))
        half = 0.6 * span + 1e-4
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

    def update(self, points_by_color, current_by_color, mode_text='', pose_t=None, pose_R=None,
               pose_history=None, pose_rot_history=None, rigid_axis_len=0.03):
        if not self.ready:
            return

        raw_pts_blocks = []
        corr_pts_blocks = []

        for name in COLOR_ORDER:
            pts = np.asarray(points_by_color.get(name, []), dtype=np.float64)
            if pts.size == 0:
                continue
            finite = np.isfinite(pts).all(axis=1)
            if not np.any(finite):
                continue

            self.lines_raw[name].set_data(pts[:, 0], pts[:, 1])
            self.lines_raw[name].set_3d_properties(pts[:, 2])
            raw_pts_blocks.append(pts)

            cur = current_by_color.get(name, None)
            if cur is not None:
                c = np.asarray(cur, dtype=np.float64).reshape(1, 3)
            else:
                idx_last = np.where(finite)[0][-1]
                c = pts[idx_last:idx_last + 1, :]
            self.current_raw[name]._offsets3d = (c[:, 0], c[:, 1], c[:, 2])

            if self.corrected_transform is not None and self.ax_corr is not None:
                pts_corr = apply_corrected_transform(pts, self.corrected_transform)
                self.lines_corr[name].set_data(pts_corr[:, 0], pts_corr[:, 1])
                self.lines_corr[name].set_3d_properties(pts_corr[:, 2])
                c_corr = apply_corrected_transform(c, self.corrected_transform)
                self.current_corr[name]._offsets3d = (c_corr[:, 0], c_corr[:, 1], c_corr[:, 2])
                corr_pts_blocks.append(pts_corr)

        axis_len = max(1e-3, float(rigid_axis_len))

        if pose_history is not None and len(pose_history) > 0:
            pose_hist_arr = np.asarray(pose_history, dtype=np.float64)
            if pose_hist_arr.ndim == 2 and pose_hist_arr.shape[1] >= 3:
                pose_hist_xyz = pose_hist_arr[:, :3]
                self.rigid_center_line_raw.set_data(pose_hist_xyz[:, 0], pose_hist_xyz[:, 1])
                self.rigid_center_line_raw.set_3d_properties(pose_hist_xyz[:, 2])
                raw_pts_blocks.append(pose_hist_xyz)
                if self.corrected_transform is not None and self.ax_corr is not None:
                    pose_hist_corr = apply_corrected_transform(pose_hist_xyz, self.corrected_transform)
                    self.rigid_center_line_corr.set_data(pose_hist_corr[:, 0], pose_hist_corr[:, 1])
                    self.rigid_center_line_corr.set_3d_properties(pose_hist_corr[:, 2])
                    corr_pts_blocks.append(pose_hist_corr)

        hist_segs_raw = self._build_axis_segments(pose_history, pose_rot_history, axis_len)
        for axis in ['x', 'y', 'z']:
            self.rigid_axes_hist_raw[axis].set_segments(hist_segs_raw[axis])

        if self.corrected_transform is not None and self.ax_corr is not None:
            for axis in ['x', 'y', 'z']:
                corr_segs = []
                for seg in hist_segs_raw[axis]:
                    seg_corr = apply_corrected_transform(seg, self.corrected_transform)
                    if np.isfinite(seg_corr).all():
                        corr_segs.append(seg_corr)
                self.rigid_axes_hist_corr[axis].set_segments(corr_segs)
        else:
            for axis in ['x', 'y', 'z']:
                self.rigid_axes_hist_corr[axis].set_segments([])

        if pose_t is not None and pose_R is not None:
            p0 = np.asarray(pose_t, dtype=np.float64).reshape(3)
            R = np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
            px = p0 + axis_len * R[:, 0]
            py = p0 + axis_len * R[:, 1]
            pz = p0 + axis_len * R[:, 2]

            self.rigid_origin_raw._offsets3d = ([p0[0]], [p0[1]], [p0[2]])
            self._set_line3d(self.rigid_axes_raw['x'], p0, px)
            self._set_line3d(self.rigid_axes_raw['y'], p0, py)
            self._set_line3d(self.rigid_axes_raw['z'], p0, pz)
            raw_pts_blocks.append(np.vstack([p0, px, py, pz]))

            if self.corrected_transform is not None and self.ax_corr is not None:
                corr_axes = apply_corrected_transform(np.vstack([p0, px, py, pz]), self.corrected_transform)
                c0, cx, cy, cz = corr_axes
                self.rigid_origin_corr._offsets3d = ([c0[0]], [c0[1]], [c0[2]])
                self._set_line3d(self.rigid_axes_corr['x'], c0, cx)
                self._set_line3d(self.rigid_axes_corr['y'], c0, cy)
                self._set_line3d(self.rigid_axes_corr['z'], c0, cz)
                corr_pts_blocks.append(corr_axes)
        else:
            self._hide_rigid_axes()

        cam_raw = np.asarray(list(self.camera_centers.values()), dtype=np.float64) if len(self.camera_centers) > 0 else None
        self._set_equal_limits(self.ax_raw, raw_pts_blocks, camera_pts=cam_raw)

        if self.corrected_transform is not None and self.ax_corr is not None:
            cam_corr = None
            if cam_raw is not None:
                cam_corr = apply_corrected_transform(cam_raw, self.corrected_transform)
            self._set_equal_limits(self.ax_corr, corr_pts_blocks, camera_pts=cam_corr)

        if mode_text:
            self.ax_raw.set_title('Raw World Frame - %s' % mode_text)
            self.ax_corr.set_title('Corrected Frame (camera plane z=0) - %s' % mode_text)

        # Cache the current matplotlib figure as BGR image for unified OpenCV preview.
        try:
            self.canvas.draw()
            self.latest_bgr = self._canvas_to_bgr()
            self.latest_error = ''
        except Exception as ex:
            self.latest_bgr = None
            self.latest_error = str(ex)

    def get_latest_frame(self):
        if self.latest_bgr is None:
            return None
        return self.latest_bgr.copy()

    def get_latest_error(self):
        return self.latest_error

    def close(self):
        self.fig = None
        self.canvas = None
        self.ready = False
