"""
AprilTag-based temporal delay calibration for multi-camera sessions.

Public API
----------
CalibSink
    VideoSink-compatible sink that stores (wall_time, gray_thumbnail) tuples.
    Attach temporarily to any HikCaptureThread / RSCaptureThread via set_sink().

run_session_temporal_calibration(hik_threads, rs_thread, hik_serials, rs_serial,
                                  output_dir, ...)
    Full calibration flow for an already-running camera session.  Attaches
    CalibSinks, shows the AprilTag fullscreen on the host monitor, detects the
    tag appearance in each stream, computes inter-camera offsets, and saves the
    result to <output_dir>/camera_delay_calib.json.

detect_tag_offsets(frames_by_serial, tag_family, tag_id, min_consecutive,
                   reference_serial)
    Lower-level helper: given a dict {serial: [(wall_time, gray), ...]}, return
    the offset dict without doing any I/O.

Standalone helper functions (also used by tools/calibrate_temporal_delay.py)
    build_detector, generate_tag_image, show_tag_on_screen, find_tag_appearance
"""

import json
import os
import threading
import time

import cv2
import numpy as np

# ─────────────────────────── constants ───────────────────────────────────────

THUMB_WIDTH = 640   # resize frames to this width before AprilTag detection

FAMILY_TO_DICT = {
    'tag36h11': 'DICT_APRILTAG_36H11',
    'tag25h9':  'DICT_APRILTAG_25H9',
    'tag16h5':  'DICT_APRILTAG_16H5',
}


# ─────────────────────────── CalibSink ───────────────────────────────────────

class CalibSink:
    """
    Lightweight VideoSink-compatible object that captures grayscale thumbnails
    for AprilTag detection.

    Compatible with both HikCaptureThread and RSCaptureThread:
        sink.submit(frame_bgr, capture_time, device_frame_num=-1)
    """

    def __init__(self, thumb_width=THUMB_WIDTH):
        self._thumb_width = int(thumb_width)
        self._frames = []          # list of (wall_time_s, gray_ndarray)
        self._lock = threading.Lock()

    def submit(self, frame_bgr, capture_time, device_frame_num=-1):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        w = gray.shape[1]
        if w > self._thumb_width:
            scale = self._thumb_width / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        with self._lock:
            self._frames.append((float(capture_time), gray))

    def get_frames(self):
        """Return a snapshot of collected (wall_time, gray) pairs."""
        with self._lock:
            return list(self._frames)

    def clear(self):
        with self._lock:
            self._frames.clear()


# ─────────────────────── AprilTag detector builder ───────────────────────────

def build_detector(family):
    """
    Return a callable  detect(gray_uint8) -> list[int]  of detected tag IDs.
    Raises RuntimeError if opencv-contrib-python is not installed.
    """
    dict_name = FAMILY_TO_DICT.get(family.lower().replace('-', ''))
    if dict_name is None:
        raise ValueError(
            'Unknown tag family %r. Choose from: %s' % (family, list(FAMILY_TO_DICT))
        )

    aruco_mod = getattr(cv2, 'aruco', None)
    if aruco_mod is None:
        raise RuntimeError(
            'cv2.aruco not found. Install opencv-contrib-python:\n'
            '  pip install opencv-contrib-python'
        )

    dict_id = getattr(aruco_mod, dict_name, None)
    if dict_id is None:
        raise RuntimeError(
            'cv2.aruco has no constant %s. '
            'Install a newer opencv-contrib-python (>=4.2).' % dict_name
        )

    if hasattr(aruco_mod, 'ArucoDetector'):
        # OpenCV 4.7+ new API
        aruco_dict = aruco_mod.getPredefinedDictionary(dict_id)
        params = aruco_mod.DetectorParameters()
        _det = aruco_mod.ArucoDetector(aruco_dict, params)

        def _detect(gray):
            _, ids, _ = _det.detectMarkers(gray)
            return [] if ids is None else ids.flatten().tolist()
    else:
        # Legacy API
        aruco_dict = aruco_mod.Dictionary_get(dict_id)
        params = aruco_mod.DetectorParameters_create()

        def _detect(gray):
            _, ids, _ = aruco_mod.detectMarkers(gray, aruco_dict, parameters=params)
            return [] if ids is None else ids.flatten().tolist()

    return _detect


# ──────────────────────── Tag image rendering ────────────────────────────────

def generate_tag_image(family, tag_id, size_px=800):
    """Render an AprilTag as a grayscale image with a white border."""
    dict_name = FAMILY_TO_DICT.get(family.lower().replace('-', ''))
    aruco_mod = cv2.aruco
    dict_id = getattr(aruco_mod, dict_name)

    if hasattr(aruco_mod, 'getPredefinedDictionary'):
        aruco_dict = aruco_mod.getPredefinedDictionary(dict_id)
        img = aruco_dict.generateImageMarker(tag_id, size_px)
    else:
        aruco_dict = aruco_mod.Dictionary_get(dict_id)
        img = aruco_mod.drawMarker(aruco_dict, tag_id, size_px)

    pad = size_px // 8
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=255)


# ──────────────────────── Fullscreen tag display ─────────────────────────────

def show_tag_on_screen(tag_img_gray, display_s=1.5,
                       win_name='AprilTag Calibration'):
    """
    Show *tag_img_gray* fullscreen for *display_s* seconds, then go black.
    Must be called from the main thread.

    Sequence: white (0.1 s) → tag (display_s) → black.
    Returns (t_show, t_hide) wall_times.
    """
    tag_bgr = cv2.cvtColor(tag_img_gray, cv2.COLOR_GRAY2BGR)
    white = np.full_like(tag_bgr, 255)
    blank = np.zeros_like(tag_bgr)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Brief white frame so cameras see the white→tag edge
    cv2.imshow(win_name, white)
    cv2.waitKey(1)
    time.sleep(0.1)

    cv2.imshow(win_name, tag_bgr)
    cv2.waitKey(1)
    t_show = time.time()

    t_end = t_show + display_s
    while time.time() < t_end:
        cv2.waitKey(20)

    cv2.imshow(win_name, blank)
    cv2.waitKey(1)
    t_hide = time.time()

    cv2.destroyWindow(win_name)
    return t_show, t_hide


# ─────────────────────── Frame-level event detection ─────────────────────────

def find_tag_appearance(frames, detect_fn, tag_id, min_consecutive=2):
    """
    Return (wall_time, frame_idx) of the first confirmed tag appearance.

    Scans *frames* [(wall_time, gray), ...] and looks for the first run of
    at least *min_consecutive* detections.  ``tag_id < 0`` matches any tag.

    Returns (None, None) if not found.
    """
    consecutive = 0
    run_start = None

    for i, (t, gray) in enumerate(frames):
        ids = detect_fn(gray)
        found = (tag_id < 0 and len(ids) > 0) or (tag_id in ids)

        if found:
            if consecutive == 0:
                run_start = i
            consecutive += 1
            if consecutive >= min_consecutive:
                return float(frames[run_start][0]), run_start
        else:
            consecutive = 0
            run_start = None

    return None, None


# ─────────────────── Core offset computation (no I/O) ────────────────────────

def detect_tag_offsets(frames_by_serial, tag_family, tag_id,
                       min_consecutive=2, reference_serial=None):
    """
    Given ``{serial: [(wall_time, gray), ...]}`` collected from each camera,
    return ``(offsets_s, event_times, reference_serial)`` or raise ValueError
    if fewer than 2 cameras detected the tag.

    ``offsets_s[serial] = event_time[serial] - event_time[reference]``
    """
    detect_fn = build_detector(tag_family)
    event_times = {}

    for serial, frames in frames_by_serial.items():
        if not frames:
            continue
        t_event, idx = find_tag_appearance(frames, detect_fn, tag_id, min_consecutive)
        if t_event is not None:
            event_times[serial] = t_event

    if len(event_times) < 2:
        raise ValueError(
            'AprilTag detected in only %d camera(s) (need ≥2). '
            'Was the tag clearly visible to all cameras?' % len(event_times)
        )

    ref = reference_serial
    if ref is None or ref not in event_times:
        ref = next(iter(event_times))

    t_ref = event_times[ref]
    offsets = {s: round(t - t_ref, 6) for s, t in event_times.items()}
    return offsets, event_times, ref


# ─────────────────── Session-level calibration entry point ───────────────────

def run_session_temporal_calibration(
    hik_threads,
    rs_thread,
    hik_serials,
    rs_serial,
    output_dir,
    tag_family='tag36h11',
    tag_id=0,
    tag_delay_s=3.0,
    tag_display_s=1.5,
    min_consecutive=2,
    reference_serial=None,
):
    """
    Run AprilTag temporal calibration using already-running camera threads.

    Attaches CalibSinks to each thread, shows the AprilTag fullscreen on the
    host monitor, detects it in each stream, computes offsets, then saves
    ``camera_delay_calib.json`` inside *output_dir*.

    Parameters
    ----------
    hik_threads     : list of HikCaptureThread (already running)
    rs_thread       : RSCaptureThread or None
    hik_serials     : list[str] — serial numbers matching hik_threads order
    rs_serial       : str or None
    output_dir      : session root directory to write the JSON result
    tag_family      : AprilTag family string (default 'tag36h11')
    tag_id          : specific tag ID to detect; -1 = any tag
    tag_delay_s     : seconds to wait before showing the tag (cameras need time
                      to stabilise and operator to aim at screen)
    tag_display_s   : seconds to keep the tag visible on screen
    min_consecutive : number of consecutive detection hits to confirm appearance
    reference_serial: serial to use as time reference (default: first hik serial)

    Returns
    -------
    dict with keys ``reference_serial``, ``offsets_s``, ``note``; or None if
    calibration was skipped due to detection failure.
    """
    print('[temporal-calib] Building AprilTag detector ...')
    tag_img = generate_tag_image(tag_family, max(0, tag_id))

    # ── Attach CalibSinks ──────────────────────────────────────────────────
    hik_sinks = [CalibSink() for _ in hik_threads]
    for th, s in zip(hik_threads, hik_sinks):
        th.set_sink(s)

    rs_sink = None
    if rs_thread is not None:
        rs_sink = CalibSink()
        rs_thread.set_sink(rs_sink)

    # ── Wait for operator to aim cameras, then countdown ─────────────────
    print()
    print('[temporal-calib] ┌──────────────────────────────────────────────────────┐')
    print('[temporal-calib] │  TEMPORAL DELAY CALIBRATION                          │')
    print('[temporal-calib] │  Aim ALL cameras at this screen, then press Enter.   │')
    print('[temporal-calib] │  The AprilTag will appear fullscreen %ds after Enter. │' % int(tag_delay_s))
    print('[temporal-calib] │  Keep cameras aimed until the screen goes black.     │')
    print('[temporal-calib] └──────────────────────────────────────────────────────┘')
    print()
    input('[temporal-calib]  Cameras aimed? Press Enter to start countdown ... ')
    print()

    for i in range(int(tag_delay_s), 0, -1):
        print('\r[temporal-calib]  Showing tag in %ds ...' % i, end='', flush=True)
        time.sleep(1.0)
    print()

    # ── Show tag on screen ────────────────────────────────────────────────
    print('[temporal-calib] Showing AprilTag ...')
    show_tag_on_screen(tag_img, display_s=tag_display_s)
    print('[temporal-calib] Tag hidden. Collecting final frames ...')
    time.sleep(0.5)

    # ── Detach sinks ──────────────────────────────────────────────────────
    for th in hik_threads:
        th.set_sink(None)
    if rs_thread is not None:
        rs_thread.set_sink(None)

    # ── Report frame counts and detect ────────────────────────────────────
    print()
    all_serials = list(hik_serials) + ([rs_serial] if rs_serial else [])
    all_sinks   = list(hik_sinks)   + ([rs_sink]   if rs_sink   else [])

    frames_by_serial = {}
    for serial, sink in zip(all_serials, all_sinks):
        frames = sink.get_frames()
        n = len(frames)
        if n > 1:
            span = frames[-1][0] - frames[0][0]
            fps_actual = (n - 1) / span if span > 0 else 0.0
            print('[temporal-calib]  [%s]  %d frames @ %.1f fps' % (serial, n, fps_actual))
        else:
            print('[temporal-calib]  [%s]  %d frames' % (serial, n))
        if n > 0:
            frames_by_serial[serial] = frames

    print()
    print('[temporal-calib] Detecting tag appearances ...')

    try:
        offsets, event_times, ref = detect_tag_offsets(
            frames_by_serial, tag_family, tag_id, min_consecutive,
            reference_serial=reference_serial or (hik_serials[0] if hik_serials else None),
        )
    except ValueError as exc:
        print('[temporal-calib] WARNING: %s' % exc)
        print('[temporal-calib] Calibration skipped — continuing without temporal offsets.')
        return None

    print('[temporal-calib] Reference: %s' % ref)
    for serial, off in offsets.items():
        label = '  <-- reference' if serial == ref else ''
        print('[temporal-calib]   %s:  %+.2f ms%s' % (serial, off * 1000.0, label))

    calib = {
        'reference_serial': ref,
        'offsets_s': offsets,
        'tag_family': tag_family,
        'tag_id': tag_id,
        'note': (
            'aligned_time = capture_wall_time - offsets_s[serial]. '
            'A positive offset means that camera captured the tag later than '
            'the reference.'
        ),
    }

    out_path = os.path.join(output_dir, 'camera_delay_calib.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=2)
    print('[temporal-calib] Saved: %s' % out_path)
    print()

    return calib
