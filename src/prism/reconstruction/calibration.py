import os
import sys


DEFAULT_LEGACY_RECORDING_DIR = '/opt/MVS/Samples/64/Python/General/Recording'


def _ensure_legacy_recording_path():
    legacy_dir = os.environ.get('PRISM_LEGACY_RECORDING_DIR', DEFAULT_LEGACY_RECORDING_DIR)
    if legacy_dir and legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)


def _legacy_led_module():
    _ensure_legacy_recording_path()
    import LedRigidBody6D4CamHSV3Color as legacy_led
    return legacy_led


def load_calibration(path):
    return _legacy_led_module().load_calibration(path)


def get_camera_centers_world(cameras):
    return _legacy_led_module().get_camera_centers_world(cameras)


def build_corrected_transform(cameras, camera_centers):
    return _legacy_led_module().build_corrected_transform(cameras, camera_centers)


def get_live_3d_plotter_class():
    return _legacy_led_module().Live3DPlotter
