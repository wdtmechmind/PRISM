import ctypes
import threading
import time
from collections import deque

from prism.common.timebase import FpsMeter

# ctypes handle to default C library; usleep() releases the GIL while sleeping
_libc = ctypes.CDLL(None)


class SoftwareTriggerThread(threading.Thread):
    """
    Fires software trigger pulses at a fixed rate.

    Design:
      Phase 1 – ctypes usleep() for most of each interval.
                usleep is a C call so Python GIL is released automatically,
                letting UI and grab threads run freely.  Unlike time.sleep(),
                it avoids the expensive setswitchinterval overhead.
      Phase 2 – pure spin for the final 0.5 ms for precise edge timing.
                0.5 ms GIL hold out of 3.33 ms interval = ~15 % GIL time.

    If the OS scheduler overshoots and we are already past next_t when Phase 1
    ends, we skip Phase 2, fire immediately, then reset next_t to `now` to
    avoid burst-firing the camera with back-to-back triggers faster than its
    minimum inter-trigger time.
    """

    def __init__(self, grabber, fps):
        super(SoftwareTriggerThread, self).__init__(daemon=True)
        self.grabber = grabber
        self.interval = 1.0 / fps
        self.stop_flag = threading.Event()

    def run(self):
        next_t = time.perf_counter()
        interval = self.interval
        while not self.stop_flag.is_set():
            # Phase 1: ctypes sleep releases GIL so other threads can run
            remaining = next_t - time.perf_counter()
            if remaining > 0.0005:
                sleep_us = int((remaining - 0.0005) * 1_000_000)
                _libc.usleep(ctypes.c_uint(max(1, sleep_us)))
            # Phase 2: spin last 0.5 ms for precise timing
            while time.perf_counter() < next_t:
                pass
            self.grabber.software_trigger_once()
            next_t += interval
            # If scheduler jitter caused us to overshoot, skip ahead by one interval
            # to avoid firing a second trigger immediately after this one
            now = time.perf_counter()
            if next_t < now:
                next_t = now + interval

    def stop(self):
        self.stop_flag.set()


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
                img_bgr, frame_num = self.grabber.grab_one_bgr(timeout_ms=self.timeout_ms)
                cap_t = time.time()
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
