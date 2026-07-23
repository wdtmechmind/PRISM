#!/usr/bin/env python3
"""
PRISM ChArUco 4-Camera Hardware-Triggered Calibration Capture

This script captures synchronized images from 4 Hik cameras using hardware trigger
for internal and external parameter calibration. It is specifically designed for
the PRISM project with:
  - Master camera (DA8165486): Software trigger with GPIO Line1 output
  - Slave cameras (cam1-3): Hardware trigger on GPIO Line0
  - Live preview with real-time ChArUco detection overlay
  - Interactive capture control

Usage:
    python3 tools/prism_charuco_calibration_capture.py \\
        --output-dir ~/mvs_charuco_data \\
        --squares-x 12 --squares-y 9 \\
        --square-length-mm 15 --marker-length-mm 11.25 \\
        --live y --detect-overlay y

Reference:
    /opt/MVS/Samples/64/Python/General/Recording/ChArUco_4Cam_Calibration_Guide.md
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add PRISM modules to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Add Hikvision MVS Python SDK (MvImport) to path so MvCameraControl_class resolves
_mv_candidates = []
if os.environ.get('PRISM_MVIMPORT_DIR'):
    _mv_candidates.append(os.environ['PRISM_MVIMPORT_DIR'])
_mv_candidates.append('/opt/MVS/Samples/64/Python/MvImport')
for _p in _mv_candidates:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism.devices.cameras.mvs_camera import (
    MvCamera,
    UsbCameraGrabber,
    enumerate_usb_devices,
    parse_indices,
)
from prism.devices.cameras.highspeed_camera import HikCaptureThread, SoftwareTriggerThread


# ChArUco board parameters
CHARUCO_DICT_MAP = {
    'DICT_4X4_50': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    'DICT_4X4_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
    'DICT_5X5_100': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    'DICT_5X5_1000': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
    'DICT_6X6_250': cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
}

def parse_yes_no(raw_text):
    return raw_text.strip().lower() in ['y', 'yes', '1', 'true']


def create_charuco_board(squares_x, squares_y, square_length_mm, marker_length_mm, 
                        aruco_dict_name='DICT_5X5_1000'):
    """Create ChArUco board for calibration."""
    aruco_dict = CHARUCO_DICT_MAP.get(aruco_dict_name)
    if aruco_dict is None:
        raise ValueError(f'Unknown ArUco dictionary: {aruco_dict_name}')
    
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        squareLength=square_length_mm / 1000.0,  # Convert mm to m
        markerLength=marker_length_mm / 1000.0,
        dictionary=aruco_dict
    )
    return board


_CHARUCO_DETECTOR_CACHE = {}


def detect_charuco_corners(gray_img, board, upsample=1, min_markers=2):
    """Detect ChArUco corners in a grayscale image (OpenCV 4.7+ CharucoDetector API)."""
    if upsample > 1:
        h, w = gray_img.shape
        gray_img = cv2.resize(gray_img, (w * upsample, h * upsample),
                             interpolation=cv2.INTER_LINEAR)

    detector = _CHARUCO_DETECTOR_CACHE.get(id(board))
    if detector is None:
        detector = cv2.aruco.CharucoDetector(board)
        _CHARUCO_DETECTOR_CACHE[id(board)] = detector

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_img)

    num_markers = 0 if marker_ids is None else len(marker_ids)
    if num_markers < min_markers:
        return None, None, num_markers, 0

    num_corners = 0 if charuco_ids is None else charuco_ids.shape[0]
    return charuco_corners, charuco_ids, num_markers, num_corners


def draw_charuco_overlay(img_bgr, board, gray_img, min_corners=6):
    """Draw ChArUco detection overlay on image."""
    overlay = img_bgr.copy()
    charuco_corners, charuco_ids, num_markers, num_corners = detect_charuco_corners(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), board, upsample=1, min_markers=1
    )
    
    status_color = (0, 255, 0) if num_corners >= min_corners else (0, 0, 255)
    status_text = f"READY ({num_corners})" if num_corners >= min_corners else f"LOW ({num_corners})"
    
    if charuco_corners is not None and charuco_ids is not None and num_corners > 0:
        # CharucoDetector returns corners as (N, 2); drawDetectedCornersCharuco
        # needs a 2-channel (N, 1, 2) array with matching (N, 1) ids.
        cc = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
        ci = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
        if cc.shape[0] == ci.shape[0]:
            overlay = cv2.aruco.drawDetectedCornersCharuco(overlay, cc, ci, (0, 255, 0))
    
    # Add status text
    cv2.putText(overlay, f"markers={num_markers} corners={num_corners}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    cv2.putText(overlay, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
               status_color, 2)
    
    return overlay, num_corners >= min_corners


class CalibrationSession:
    def __init__(self, output_dir, squares_x, squares_y, square_length_mm, 
                 marker_length_mm, aruco_dict, exposure_us, gain, frame_rate):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_dir / f'CharucoCapture_{timestamp}'
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.board = create_charuco_board(squares_x, squares_y, square_length_mm,
                                        marker_length_mm, aruco_dict)
        
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length_mm = square_length_mm
        self.marker_length_mm = marker_length_mm
        self.aruco_dict = aruco_dict
        
        self.exposure_us = exposure_us
        self.gain = gain
        self.frame_rate = frame_rate
        
        self.cameras = []
        self.capture_threads = []
        self.trigger_thread = None
        self.master_cam_index = None
        self.frame_count = 0
        
        # Master camera serial
        self.master_serial = 'DA8165486'
        
        print(f'[CalibrationSession] output: {self.session_dir}')

    def initialize_cameras(self):
        """Initialize and open all 4 cameras with hardware trigger."""
        MvCamera.MV_CC_Initialize()
        print(f'SDKVersion[0x{MvCamera.MV_CC_GetSDKVersion():x}]')
        
        usb_devices = enumerate_usb_devices()
        if len(usb_devices) < 4:
            raise RuntimeError(f'Found {len(usb_devices)} USB cameras, need 4 Hik cameras')
        
        print(f'Found {len(usb_devices)} USB cameras:')
        for i, _, model, serial in usb_devices:
            print(f'  [{i}] model={model}, serial={serial}')
        
        # Select first 4 or prompt user
        selected_text = input('Please input 4 Hik camera indices (blank=0,1,2,3): ').strip()
        selected = list(range(4)) if selected_text == '' else parse_indices(selected_text, len(usb_devices), 4)
        
        selected_infos = [usb_devices[idx] for idx in selected]
        
        for cam_i, (_, dev_info, model, serial) in enumerate(selected_infos):
            serial_safe = serial if serial else f'cam{cam_i}'
            print(f'\nInitializing hik{cam_i}: model={model} serial={serial_safe}')
            
            cam = UsbCameraGrabber(dev_info, serial_safe, model)
            
            if serial == self.master_serial:
                self.master_cam_index = cam_i
                print('  -> Master camera: software trigger + GPIO Line1 output')
                cam.open_and_prepare(
                    exposure_us=self.exposure_us,
                    gain=self.gain,
                    frame_rate=self.frame_rate,
                    trigger_source='Software',
                    gpio_output_line=1,
                )
            else:
                print('  -> Slave camera: hardware trigger on GPIO Line0')
                cam.open_and_prepare(
                    exposure_us=self.exposure_us,
                    gain=self.gain,
                    frame_rate=self.frame_rate,
                    trigger_source='Line0',
                    gpio_output_line=None,
                )
            
            self.cameras.append((cam_i, cam, serial_safe))
    
    def start_streaming(self):
        """Start per-camera capture threads and the master software trigger.

        The master camera runs in software-trigger mode and drives the slaves
        through its GPIO strobe, so nothing is captured until this trigger
        thread is running.
        """
        self.capture_threads = [None] * len(self.cameras)
        for cam_i, cam, serial in self.cameras:
            th = HikCaptureThread(cam, cam_i, serial, timeout_ms=1000, buffer_len=5)
            th.start()
            self.capture_threads[cam_i] = th

        if self.master_cam_index is not None:
            master_cam = self.cameras[self.master_cam_index][1]
            self.trigger_thread = SoftwareTriggerThread(master_cam, fps=self.frame_rate)
            self.trigger_thread.start()
            print('Master software trigger started at %.1f fps' % self.frame_rate)
        else:
            print('WARNING: master camera %s not found; no trigger will fire and no frames will arrive.'
                  % self.master_serial)

    def stop_streaming(self):
        """Stop the software trigger and all capture threads."""
        if self.trigger_thread is not None:
            try:
                self.trigger_thread.stop()
                self.trigger_thread.join(timeout=1.0)
            except Exception as e:
                print('Error stopping trigger thread: %s' % e)
            self.trigger_thread = None

        for th in self.capture_threads:
            if th is not None:
                th.stop()
        for th in self.capture_threads:
            if th is not None:
                th.join(timeout=2.0)
        self.capture_threads = []

    def _find_cam_index(self, serial):
        """Find camera index by serial."""
        for cam_i, _, s in self.cameras:
            if s == serial:
                return cam_i
        return -1

    def capture_current(self, min_corners=6):
        """Save the latest live frame from every camera as one synchronized set."""
        frames = {}
        for cam_i, cam, serial in self.cameras:
            th = self.capture_threads[cam_i] if cam_i < len(self.capture_threads) else None
            frame = th.get_latest() if th is not None else None
            if frame is None:
                print('  cam%d (%s): no frame yet, capture aborted' % (cam_i, serial))
                return False
            frames[serial] = frame.copy()

        frame_id = self.frame_count
        for serial, img_bgr in frames.items():
            cam_dir = self.session_dir / ('cam%d_%s' % (self._find_cam_index(serial), serial))
            cam_dir.mkdir(parents=True, exist_ok=True)
            img_path = cam_dir / ('frame_%04d.png' % frame_id)
            cv2.imwrite(str(img_path), img_bgr)
        self.frame_count += 1
        print('captured frame set %d (%d cameras) -> %s' % (frame_id, len(frames), self.session_dir))
        return True

    def _build_preview_cell(self, cam_i, serial, frame, preview_width, min_corners):
        """Return a preview cell (BGR) with ChArUco overlay and camera label."""
        if frame is None:
            cell = np.zeros((int(preview_width * 0.75), preview_width, 3), dtype=np.uint8)
            cv2.putText(cell, 'cam%d %s waiting...' % (cam_i, serial),
                        (10, cell.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 200, 200), 2, cv2.LINE_AA)
            return cell, False

        overlay, ready = draw_charuco_overlay(frame, self.board, None, min_corners=min_corners)

        label = 'cam%d %s%s' % (cam_i, serial, ' [MASTER]' if serial == self.master_serial else '')
        cv2.putText(overlay, label, (10, overlay.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        h, w = overlay.shape[:2]
        scale = preview_width / float(w)
        cell = cv2.resize(overlay, (preview_width, max(1, int(round(h * scale)))))
        return cell, ready

    @staticmethod
    def _tile_2x2(cells):
        """Tile up to 4 equal-sized cells into a 2x2 grid, padding as needed."""
        h = max(c.shape[0] for c in cells)
        w = max(c.shape[1] for c in cells)
        padded = []
        for c in cells:
            ph, pw = c.shape[:2]
            if ph != h or pw != w:
                c = cv2.copyMakeBorder(c, 0, h - ph, 0, w - pw, cv2.BORDER_CONSTANT, value=0)
            padded.append(c)
        while len(padded) < 4:
            padded.append(np.zeros((h, w, 3), dtype=np.uint8))
        top = np.hstack([padded[0], padded[1]])
        bottom = np.hstack([padded[2], padded[3]])
        return np.vstack([top, bottom])

    def run_live_capture(self, preview_width=480, min_corners=6):
        """Live 2x2 preview with real-time ChArUco overlay; capture on keypress."""
        win = 'PRISM ChArUco Calibration'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, preview_width * 2, int(preview_width * 0.75) * 2 + 40)

        print('\n[Live Mode]')
        print('  Adjust the ChArUco board while watching the overlay.')
        print('  Green corners = detected. Aim for all 4 cameras showing READY.')
        print('  Keys: SPACE / ENTER = capture frame set, q / ESC = finish.')

        while True:
            cells = []
            ready_flags = []
            for cam_i, cam, serial in self.cameras:
                th = self.capture_threads[cam_i] if cam_i < len(self.capture_threads) else None
                frame = th.get_latest() if th is not None else None
                cell, ready = self._build_preview_cell(cam_i, serial, frame, preview_width, min_corners)
                cells.append(cell)
                ready_flags.append(ready)

            if not cells:
                break

            grid = self._tile_2x2(cells)

            ready_count = sum(1 for r in ready_flags if r)
            all_ready = ready_count == len(self.cameras)
            header_color = (0, 255, 0) if all_ready else (0, 200, 255)
            cv2.putText(grid,
                        'captured=%d | cams ready=%d/%d%s' % (
                            self.frame_count, ready_count, len(self.cameras),
                            '  <-- ALL READY, press SPACE' if all_ready else ''),
                        (10, grid.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        header_color, 2, cv2.LINE_AA)

            cv2.imshow(win, grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord(' '), 10, 13):
                self.capture_current(min_corners=min_corners)

        cv2.destroyWindow(win)
        print('\nCaptured %d frame sets.' % self.frame_count)

    def run_terminal_capture(self, min_corners=6):
        """Headless capture: streaming runs in background, ENTER saves a set."""
        print('\n[Terminal Mode] streaming is live in the background.')
        print('Press ENTER to capture a frame set, or "q" to quit.')
        while True:
            user_input = input('Frame %d: ' % self.frame_count).strip().lower()
            if user_input == 'q':
                break
            elif user_input == '':
                self.capture_current(min_corners=min_corners)
            else:
                print('Unknown command. Press ENTER to capture or "q" to quit.')
        print('\nCaptured %d frame sets.' % self.frame_count)
    
    def save_metadata(self):
        """Save calibration session metadata."""
        metadata = {
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'board': {
                'squares_x': self.squares_x,
                'squares_y': self.squares_y,
                'square_length_mm': self.square_length_mm,
                'marker_length_mm': self.marker_length_mm,
                'aruco_dict': self.aruco_dict,
            },
            'camera_settings': {
                'exposure_us': self.exposure_us,
                'gain': self.gain,
                'frame_rate': self.frame_rate,
                'trigger_mode': 'hardware_sync',
                'master_serial': self.master_serial,
            },
            'frame_count': self.frame_count,
            'trigger_config': {
                'master': 'software_trigger + GPIO_Line1_output',
                'slaves': 'hardware_trigger_on_GPIO_Line0',
            }
        }
        
        metadata_path = self.session_dir / 'calibration_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f'\nMetadata saved to: {metadata_path}')
        return metadata_path
    
    def cleanup(self):
        """Stop streaming and close all cameras."""
        self.stop_streaming()
        for _, cam, _ in self.cameras:
            try:
                cam.stop_and_close()
            except Exception as e:
                print(f'Error closing camera: {e}')
        
        MvCamera.MV_CC_Finalize()
        print('Cameras closed.')


def main():
    parser = argparse.ArgumentParser(
        description='PRISM ChArUco 4-Camera Hardware-Triggered Calibration Capture',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--output-dir', type=str, default='~/mvs_charuco_data',
                       help='Output directory for calibration frames')
    parser.add_argument('--squares-x', type=int, default=12,
                       help='Number of squares in X direction')
    parser.add_argument('--squares-y', type=int, default=9,
                       help='Number of squares in Y direction')
    parser.add_argument('--square-length-mm', type=float, default=15.0,
                       help='Size of each square (mm)')
    parser.add_argument('--marker-length-mm', type=float, default=11.25,
                       help='Size of ArUco marker (mm)')
    parser.add_argument('--aruco-dict', type=str, default='DICT_5X5_1000',
                       help='ArUco dictionary')
    parser.add_argument('--exposure-us', type=float, default=12000.0,
                       help='Camera exposure time (microseconds)')
    parser.add_argument('--gain', type=float, default=0.0,
                       help='Camera gain')
    parser.add_argument('--frame-rate', type=float, default=15.0,
                       help='Acquisition frame rate (fps)')
    parser.add_argument('--live', type=str, default='y',
                       help='Enable live preview (y/n)')
    parser.add_argument('--detect-overlay', type=str, default='n',
                       help='Enable ChArUco detection overlay (y/n)')
    parser.add_argument('--min-corners', type=int, default=6,
                       help='Minimum corners for "READY" status')
    parser.add_argument('--preview-width', type=int, default=480,
                       help='Preview image width')
    
    args = parser.parse_args()
    
    session = CalibrationSession(
        output_dir=args.output_dir,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
        aruco_dict=args.aruco_dict,
        exposure_us=args.exposure_us,
        gain=args.gain,
        frame_rate=args.frame_rate,
    )
    
    try:
        print('Initializing cameras...')
        session.initialize_cameras()

        print('\nStarting live streaming (master software trigger)...')
        session.start_streaming()

        if parse_yes_no(args.live):
            session.run_live_capture(
                preview_width=args.preview_width,
                min_corners=args.min_corners,
            )
        else:
            session.run_terminal_capture(min_corners=args.min_corners)

        session.save_metadata()
        
        print(f'\n✓ Calibration frames saved to: {session.session_dir}')
        print(f'Next step: Run calibration with CharucoCalibrate4Cam.py')
        print(f'Example:')
        print(f'  python3 /opt/MVS/Samples/64/Python/General/Recording/CharucoCalibrate4Cam.py \\')
        print(f'    --dataset-root {session.session_dir} \\')
        print(f'    --output ~/mvs_charuco_data/charuco_4cam_result.json')
        
    finally:
        session.cleanup()


if __name__ == '__main__':
    main()
