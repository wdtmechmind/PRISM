import csv
import os
import queue
import threading

import cv2
import numpy as np


def brighten_bgr(img, alpha, beta):
    if abs(alpha - 1.0) < 1e-9 and abs(beta) < 1e-9:
        return img
    return cv2.convertScaleAbs(img, alpha=float(alpha), beta=float(beta))


def make_ffmpeg_safe_fps(fps, default=30.0):
    """Quantize fps so ffmpeg/mpeg4 timebase denominator stays <= 65535."""
    try:
        f = float(fps)
    except Exception:
        return float(default)
    if not np.isfinite(f) or f < 1.0:
        return float(default)

    for precision in (2, 1, 0):
        scale = 10 ** precision
        num = int(round(f * scale))
        if 1 <= num <= 65535:
            return num / float(scale)

    return float(int(min(65535, max(1, round(f)))))


class VideoSink(object):
    def __init__(self, path, fps, max_queue=256, brightness_alpha=1.0, brightness_beta=0.0,
                 trial_start_wall=None):
        self.path = path
        self.fps = make_ffmpeg_safe_fps(fps, default=30.0)
        self.q = queue.Queue(maxsize=int(max_queue))
        self.writer = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.stop_flag = threading.Event()
        self.writer_failed = False
        self.dropped = 0
        self.written = 0
        self.brightness_alpha = float(brightness_alpha)
        self.brightness_beta = float(brightness_beta)
        # Reference wall time for the trial-relative time axis; None -> column left blank.
        self.trial_start_wall = float(trial_start_wall) if trial_start_wall is not None else None

        ts_path = os.path.splitext(path)[0] + '_timestamps.csv'
        self.ts_file = open(ts_path, 'w', newline='', encoding='utf-8')
        self.ts_writer = csv.writer(self.ts_file)
        self.ts_writer.writerow(['frame_index', 'capture_wall_time', 'trial_time', 'device_frame_num'])

        self.thread.start()

    def _ensure_writer(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (int(w), int(h)))
            if not self.writer.isOpened():
                raise RuntimeError('failed to open video writer: %s' % self.path)

    def submit(self, frame, capture_time, device_frame_num=-1):
        try:
            self.q.put_nowait((frame, float(capture_time), int(device_frame_num)))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not (self.stop_flag.is_set() and self.q.empty()):
            try:
                frame, capture_time, device_frame_num = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.writer_failed:
                self.dropped += 1
                continue
            if self.brightness_alpha != 1.0 or self.brightness_beta != 0.0:
                frame = brighten_bgr(frame, self.brightness_alpha, self.brightness_beta)
            try:
                self._ensure_writer(frame)
                self.writer.write(frame)
                if self.trial_start_wall is not None:
                    trial_time = '%.6f' % (capture_time - self.trial_start_wall)
                else:
                    trial_time = ''
                self.ts_writer.writerow([self.written, '%.6f' % capture_time, trial_time, device_frame_num])
                self.written += 1
            except Exception as e:
                self.writer_failed = True
                print('warning: disable writer %s (%s)' % (self.path, str(e)))

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=10.0)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.ts_file is not None:
            self.ts_file.flush()
            self.ts_file.close()
            self.ts_file = None
