import threading
import time
from collections import deque

import cv2

from prism.common.timebase import FpsMeter


class HikCaptureThread(threading.Thread):
    def __init__(self, grabber, cam_index, serial, timeout_ms=200, buffer_len=15):
        super(HikCaptureThread, self).__init__(daemon=True)
        self.grabber = grabber
        self.cam_index = cam_index
        self.serial = serial
        self.timeout_ms = int(timeout_ms)
        self.stop_flag = threading.Event()
        self.latest = None
        self.latest_lock = threading.Lock()
        self.fps_meter = FpsMeter(window=90)
        self.sink = None
        self.sink_lock = threading.Lock()
        self.grabbed = 0
        self.buffer = deque(maxlen=int(buffer_len))
        self.buffer_lock = threading.Lock()

    def set_sink(self, sink):
        with self.sink_lock:
            self.sink = sink

    def get_latest(self):
        with self.latest_lock:
            if self.latest is None:
                return None
            return self.latest

    def get_buffer(self):
        with self.buffer_lock:
            return list(self.buffer)

    def run(self):
        while not self.stop_flag.is_set():
            try:
                img_rgb, frame_num = self.grabber.grab_one_rgb(timeout_ms=self.timeout_ms)
                cap_t = time.time()
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                continue

            self.grabbed += 1
            self.fps_meter.tick(cap_t)

            with self.latest_lock:
                self.latest = img_bgr

            with self.buffer_lock:
                self.buffer.append((cap_t, img_bgr))

            with self.sink_lock:
                sink = self.sink
            if sink is not None:
                sink.submit(img_bgr.copy(), cap_t, frame_num)

    def stop(self):
        self.stop_flag.set()
