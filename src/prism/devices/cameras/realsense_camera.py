import json
import threading
import time
from collections import deque

import cv2
import numpy as np

from prism.common.timebase import FpsMeter

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


def load_rs_intrinsics(calib_json_path):
    with open(calib_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    intr = data['intrinsics']
    K = np.asarray(intr['K'], dtype=np.float64).reshape(3, 3)
    D = np.asarray(intr['D'], dtype=np.float64).reshape(-1)

    size = data.get('image_size', None)
    if size is not None:
        calib_wh = (int(size[0]), int(size[1]))
    else:
        calib_wh = None
    return K, D, calib_wh, data


def build_undistort_maps(K, D, width, height):
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0.0, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (width, height), cv2.CV_16SC2)
    return map1, map2, new_K


class RealSenseColorGrabber(object):
    def __init__(
        self,
        serial=None,
        width=1280,
        height=720,
        fps=30,
        auto_exposure=False,
        exposure=None,
        gain=None,
        brightness=None,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.auto_exposure = bool(auto_exposure)
        self.exposure = exposure
        self.gain = gain
        self.brightness = brightness

        self.pipeline = None
        self.profile = None
        self.color_sensor = None

    def _set_option_if_supported(self, option_name, value):
        option = getattr(rs.option, option_name)
        if self.color_sensor.supports(option):
            self.color_sensor.set_option(option, float(value))
            return True
        return False

    def _read_option_if_supported(self, option_name):
        option = getattr(rs.option, option_name)
        if self.color_sensor.supports(option):
            return self.color_sensor.get_option(option)
        return None

    def open_and_prepare(self):
        if rs is None:
            raise RuntimeError('pyrealsense2 is not available. Please install librealsense/python wrapper first.')

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(config)
        device = self.profile.get_device()
        self.color_sensor = device.first_color_sensor()

        if self.auto_exposure:
            self._set_option_if_supported('enable_auto_exposure', 1)
        else:
            self._set_option_if_supported('enable_auto_exposure', 0)
            if self.exposure is not None:
                self._set_option_if_supported('exposure', self.exposure)
            if self.gain is not None:
                self._set_option_if_supported('gain', self.gain)

        if self.brightness is not None:
            self._set_option_if_supported('brightness', self.brightness)

        for _ in range(10):
            self.pipeline.wait_for_frames()

        readback = {
            'enable_auto_exposure': self._read_option_if_supported('enable_auto_exposure'),
            'exposure': self._read_option_if_supported('exposure'),
            'gain': self._read_option_if_supported('gain'),
            'brightness': self._read_option_if_supported('brightness'),
        }
        print('realsense readback: %s' % readback)
        return readback

    def grab_color_bgr(self, timeout_ms=1000):
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        img = np.asanyarray(color_frame.get_data())
        if img is None or img.size == 0:
            return None
        return img

    def poll_color_bgr(self):
        if self.pipeline is None:
            return None, None
        frames = self.pipeline.poll_for_frames()
        if not frames:
            return None, None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None, None
        img = np.asanyarray(color_frame.get_data())
        if img is None or img.size == 0:
            return None, None
        return img, int(color_frame.get_frame_number())

    def stop_and_close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


class RSCaptureThread(threading.Thread):
    def __init__(self, rs_grabber, undistort_maps, rs_width, rs_height, timeout_ms=1000, buffer_len=10):
        super(RSCaptureThread, self).__init__(daemon=True)
        self.rs_grabber = rs_grabber
        self.undistort_maps = undistort_maps
        self.rs_width = int(rs_width)
        self.rs_height = int(rs_height)
        self.timeout_ms = int(timeout_ms)
        self.stop_flag = threading.Event()
        self.latest = None
        self.latest_lock = threading.Lock()
        self.fps_meter = FpsMeter(window=60)
        self.sink = None
        self.sink_lock = threading.Lock()
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
            img = self.rs_grabber.grab_color_bgr(timeout_ms=self.timeout_ms)
            cap_t = time.time()
            if img is None:
                continue
            if self.undistort_maps is not None and img.shape[1] == self.rs_width and img.shape[0] == self.rs_height:
                img = cv2.remap(img, self.undistort_maps[0], self.undistort_maps[1], cv2.INTER_LINEAR)

            self.fps_meter.tick(cap_t)
            with self.latest_lock:
                self.latest = img

            with self.buffer_lock:
                self.buffer.append((cap_t, img))

            with self.sink_lock:
                sink = self.sink
            if sink is not None:
                sink.submit(img.copy(), cap_t)

    def stop(self):
        self.stop_flag.set()
