"""
RPi GPIO interface for 5-channel magnetic encoder PWM reading, LED control,
and hand socket triggering.

Encoder PWM inputs:  GPIO 17, 27, 22, 5, 6  (3.3V PWM signals)
LED outputs:         GPIO 23, 24, 25, 12, 16

Uses gpiozero for GPIO control.
"""

import argparse
import glob
import json
import os
import socket
import time
import threading
from gpiozero import LED, AngularServo
from gpiozero.pins.rpigpio import RPiGPIOFactory
from gpiozero import Device
import pigpio

# --- Pin Definitions ---
ENCODER_PINS = [17, 27, 22, 5, 6]   # PWM input pins for encoders 1-5
LED_PINS     = [23, 24, 25, 12, 16]  # Output pins for LEDs 1-5
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DEFAULT_ENCODER_CALIBRATION = os.path.join(ROOT_DIR, 'configs', 'devices', 'encoder_pwm_calibration.json')
ENCODER_CALIBRATION_GLOB = os.path.join(ROOT_DIR, 'configs', 'devices', 'encoder_pwm_calibration*.json')
DEFAULT_HAND_HOST = os.environ.get('PRISM_HAND_HOST') or os.environ.get('SSH_CLIENT', '10.12.194.2').split()[0]
DEFAULT_HAND_PORT = int(os.environ.get('PRISM_HAND_PORT', '60686'))
DEFAULT_HAND_TIMEOUT_S = float(os.environ.get('PRISM_HAND_TIMEOUT_S', '0.5'))
DEFAULT_EVENT_HOST = os.environ.get('PRISM_RPI_EVENT_HOST') or DEFAULT_HAND_HOST
DEFAULT_EVENT_PORT = int(os.environ.get('PRISM_RPI_EVENT_PORT', '60701'))
DEFAULT_TRIGGER_THRESHOLD = float(os.environ.get('PRISM_HAND_TRIGGER_THRESHOLD', '0.5'))
DEFAULT_FIVE_GRASP_THRESHOLD = float(os.environ.get('PRISM_FIVE_GRASP_THRESHOLD', '0.45'))
DEFAULT_THUMB_IN_THRESHOLD = float(os.environ.get('PRISM_THUMB_IN_THRESHOLD', '0.5'))
DEFAULT_INDEX_PRESS_MIN = float(os.environ.get('PRISM_INDEX_PRESS_MIN', '0.2'))
DEFAULT_INDEX_CLICK_WINDOW_S = float(os.environ.get('PRISM_INDEX_CLICK_WINDOW_S', '0.7'))

GESTURE_COMMANDS = {
    'five_grasp': '@ROG<0>&',
    'five_open': '@ROG<1>&',
    'two_grasp_b': '@ROG<4>&',
    'two_open_b': '@ROG<5>&',
    'three_grasp_b': '@ROG<8>&',
    'three_open_b': '@ROG<9>&',
    'thumb_in': '@ROG<11>&',
    'thumb_out': '@ROG<12>&',
    'index_point': '@ROG<13>&',
    'index_press': '@ROG<14>&',
    'index_single_click': '@ROG<15>&',
    'index_double_click': '@ROG<16>&',
}

RECOVERY_POSES = {
    'five_grasp': ('five_open', 'thumb_out'),
    'two_grasp_b': 'two_open_b',
    'three_grasp_b': ('three_open_b', 'thumb_out'),
    'index_point': ('five_open', 'thumb_out'),
    'index_press': ('five_open', 'thumb_out'),
    'thumb_in': 'thumb_out',
}

EXCLUSIVE_ACTIONS = ('five_grasp', 'three_grasp_b', 'index_point', 'index_press')
ACTIVE_ACTION_ORDER = ('two_grasp_b', 'thumb_in', 'five_grasp', 'three_grasp_b', 'index_point', 'index_press')

# --- PWM Reception via pigpio ---
# gpiozero does not natively read PWM duty cycle/frequency.
# We use pigpio callbacks to measure pulse widths for encoder angle decoding.

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running. Run: sudo pigpiod")

# Storage for latest PWM measurements per encoder pin.
_pulse_widths_us = {pin: 0 for pin in ENCODER_PINS}
_periods_us = {pin: 0 for pin in ENCODER_PINS}
_duties = {pin: None for pin in ENCODER_PINS}
_encoder_calibration = None
_encoder_calibration_path = None
_hand_host = DEFAULT_HAND_HOST
_hand_port = DEFAULT_HAND_PORT
_hand_timeout_s = DEFAULT_HAND_TIMEOUT_S
_event_host = DEFAULT_EVENT_HOST
_event_port = DEFAULT_EVENT_PORT
_event_logging_enabled = True
_hand_trigger_threshold = DEFAULT_TRIGGER_THRESHOLD
_five_grasp_threshold = DEFAULT_FIVE_GRASP_THRESHOLD
_thumb_in_threshold = DEFAULT_THUMB_IN_THRESHOLD
_index_press_min = DEFAULT_INDEX_PRESS_MIN
_index_click_window_s = DEFAULT_INDEX_CLICK_WINDOW_S
_hand_trigger_enabled = True
_active_hand_actions = set()
_index_click_was_above = False
_last_index_click_time = None
_callbacks = []

def _make_pwm_callback(pin):
    """Create a pigpio callback to measure PWM high width, period, and duty."""
    _last_tick = [None]

    def _cb(gpio, level, tick):
        if level == 1:
            if _last_tick[0] is not None:
                period_us = pigpio.tickDiff(_last_tick[0], tick)
                if period_us > 0:
                    _periods_us[gpio] = period_us
            _last_tick[0] = tick
        elif level == 0 and _last_tick[0] is not None:
            pulse_width = pigpio.tickDiff(_last_tick[0], tick)
            _pulse_widths_us[gpio] = pulse_width
            period_us = _periods_us[gpio]
            if period_us > 0 and 0 < pulse_width <= period_us:
                _duties[gpio] = pulse_width / period_us

    return _cb

def _setup_encoder_callbacks():
    for pin in ENCODER_PINS:
        pi.set_mode(pin, pigpio.INPUT)
        cb = pi.callback(pin, pigpio.EITHER_EDGE, _make_pwm_callback(pin))
        _callbacks.append(cb)

def pulse_width_to_angle(pulse_us, min_us=500, max_us=2500):
    """
    Convert PWM pulse width to angle in degrees.
    Assumes common 500–2500 µs range maps to 0–360°.
    Adjust min_us/max_us to match your encoder's spec.
    """
    pulse_us = max(min_us, min(max_us, pulse_us))
    return (pulse_us - min_us) / (max_us - min_us) * 360.0

def find_latest_encoder_calibration():
    """Return the newest encoder calibration JSON path, or None if unavailable."""
    candidates = glob.glob(ENCODER_CALIBRATION_GLOB)
    if os.path.isfile(DEFAULT_ENCODER_CALIBRATION) and DEFAULT_ENCODER_CALIBRATION not in candidates:
        candidates.append(DEFAULT_ENCODER_CALIBRATION)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def load_encoder_calibration(path=None):
    """Load duty-cycle calibration JSON produced by tools/calibrate_encoder_pwm.py."""
    global _encoder_calibration, _encoder_calibration_path
    if path is None:
        path = find_latest_encoder_calibration()
    if not path or not os.path.isfile(path):
        _encoder_calibration = {}
        _encoder_calibration_path = None
        return _encoder_calibration

    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    calibration = {}
    for i, pin in enumerate(ENCODER_PINS):
        channel = i + 1
        item = (data.get('encoders') or {}).get('Enc%d' % channel)
        if not item:
            continue
        min_duty = item.get('min_duty')
        max_duty = item.get('max_duty')
        if min_duty is None or max_duty is None or max_duty <= min_duty:
            continue
        calibration[channel] = {
            'pin': item.get('pin', pin),
            'min_duty': float(min_duty),
            'max_duty': float(max_duty),
            'reverse': bool(item.get('reverse', False)),
            'zero_angle_deg': float(item.get('zero_angle_deg', 0.0)),
            'angle_range_deg': float(item.get('angle_range_deg', 360.0)),
        }

    _encoder_calibration = calibration
    _encoder_calibration_path = path
    return _encoder_calibration

def duty_to_normalized_position(duty, channel):
    """Convert measured duty cycle to calibrated 0..1 mechanical position."""
    if _encoder_calibration is None:
        load_encoder_calibration()
    item = _encoder_calibration.get(channel)
    if item is None or duty is None:
        return None

    min_duty = item['min_duty']
    max_duty = item['max_duty']
    duty = max(min_duty, min(max_duty, duty))
    normalized = (duty - min_duty) / (max_duty - min_duty)
    if item['reverse']:
        normalized = 1.0 - normalized
    return normalized

def duty_to_angle(duty, channel):
    """Convert a measured duty cycle to an angle using per-channel calibration."""
    if _encoder_calibration is None:
        load_encoder_calibration()
    item = _encoder_calibration.get(channel)
    normalized = duty_to_normalized_position(duty, channel)
    if item is None or normalized is None:
        return None
    return item['zero_angle_deg'] + normalized * item['angle_range_deg']

def get_encoder_positions(use_calibration=True):
    """Return calibrated 0..1 positions for encoders with valid calibration."""
    positions = {}
    for i, pin in enumerate(ENCODER_PINS):
        channel = i + 1
        if use_calibration:
            positions[channel] = duty_to_normalized_position(_duties[pin], channel)
        else:
            positions[channel] = _duties[pin]
    return positions

def get_encoder_angles(use_calibration=True):
    """Return dict of {encoder_index: angle_degrees} for encoders 1-5."""
    angles = {}
    for i, pin in enumerate(ENCODER_PINS):
        channel = i + 1
        if use_calibration:
            angle = duty_to_angle(_duties[pin], channel)
            if angle is not None:
                angles[channel] = angle
                continue
        pw = _pulse_widths_us[pin]
        angles[channel] = pulse_width_to_angle(pw) if pw > 0 else None
    return angles

def configure_hand_trigger(host=None, port=None, timeout_s=None, threshold=None,
                           five_grasp_threshold=None, thumb_in_threshold=None,
                           index_press_min=None, index_click_window_s=None, enabled=True,
                           event_host=None, event_port=None, event_logging_enabled=True):
    """Configure edge-triggered hand socket commands from encoder state."""
    global _hand_host, _hand_port, _hand_timeout_s, _hand_trigger_threshold, _five_grasp_threshold
    global _thumb_in_threshold, _index_press_min, _index_click_window_s, _hand_trigger_enabled
    global _event_host, _event_port, _event_logging_enabled
    if host is not None:
        _hand_host = host
    if port is not None:
        _hand_port = int(port)
    if timeout_s is not None:
        _hand_timeout_s = float(timeout_s)
    if threshold is not None:
        _hand_trigger_threshold = float(threshold)
    if five_grasp_threshold is not None:
        _five_grasp_threshold = float(five_grasp_threshold)
    if thumb_in_threshold is not None:
        _thumb_in_threshold = float(thumb_in_threshold)
    if index_press_min is not None:
        _index_press_min = float(index_press_min)
    if index_click_window_s is not None:
        _index_click_window_s = float(index_click_window_s)
    if event_host is not None:
        _event_host = event_host
    if event_port is not None:
        _event_port = int(event_port)
    _event_logging_enabled = bool(event_logging_enabled)
    _hand_trigger_enabled = bool(enabled)

def send_hand_command(command):
    """Send a raw DexHand SDK command to the host hand TCP port."""
    if not _hand_host:
        raise RuntimeError('hand host is empty')

    cmd = command.strip()
    if not cmd.endswith('&'):
        cmd += '&'
    with socket.create_connection((_hand_host, int(_hand_port)), timeout=_hand_timeout_s) as sock:
        sock.sendall(cmd.encode('utf-8'))
    return cmd

def send_event_log(action, command, status='ok', message=''):
    """Send one RPi hand event to the PRISM collector over UDP."""
    if not _event_logging_enabled or not _event_host or int(_event_port) <= 0:
        return False
    positions = get_encoder_positions(use_calibration=True)
    angles = get_encoder_angles(use_calibration=True)
    readings = get_encoder_readings()
    payload = {
        'schema': 'prism.rpi_hand_event.v1',
        'wall_time': time.time(),
        'monotonic': time.monotonic(),
        'action': action,
        'command': command,
        'status': status,
        'message': message,
        'positions': positions,
        'angles': angles,
        'readings': readings,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(data, (_event_host, int(_event_port)))
    return True

def _positions_ready(positions):
    return all(positions.get(channel) is not None for channel in range(1, 6))

def _closed(value):
    return value is not None and value > _hand_trigger_threshold

def _open(value):
    return value is not None and value <= _hand_trigger_threshold

def _five_grasp_closed(value):
    return value is not None and value > _five_grasp_threshold

def _thumb_in(value):
    return value is not None and value > _thumb_in_threshold

def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return (value,)

def classify_encoder_pose(positions):
    """Map Enc1..Enc5 calibrated positions to one stable CLI hand pose.

    Enc1: thumb horizontal swing; Enc2: thumb flexion;
    Enc3: ring; Enc4: middle; Enc5: index.
    Click actions are handled separately because they are edge events.
    """
    if not _positions_ready(positions):
        return None

    thumb_swing = positions[1]
    thumb_flex = positions[2]
    ring = positions[3]
    middle = positions[4]
    index = positions[5]

    thumb_flex_closed = _closed(thumb_flex)
    ring_closed = _closed(ring)
    middle_closed = _closed(middle)
    index_closed = _closed(index)

    if thumb_flex_closed and middle_closed and ring_closed and _index_press_min <= index <= _hand_trigger_threshold:
        return 'index_press'

    if thumb_flex_closed and _open(index) and middle_closed and ring_closed:
        return 'index_point'

    if all(_five_grasp_closed(positions[channel]) for channel in (2, 3, 4)):
        return 'five_grasp'

    if middle_closed and index_closed:
        return 'three_grasp_b'

    if _thumb_in(thumb_swing):
        return 'thumb_in'

    return None

def _index_click_pose(positions):
    global _index_click_was_above, _last_index_click_time
    if not _positions_ready(positions):
        return None

    index_above = _closed(positions[5])
    click_context = (
        _open(positions[2]) and _open(positions[3]) and _open(positions[4])
    )
    pose = None
    if index_above and not _index_click_was_above and click_context:
        now = time.monotonic()
        if _last_index_click_time is not None and now - _last_index_click_time <= _index_click_window_s:
            pose = 'index_double_click'
            _last_index_click_time = None
        else:
            pose = 'index_single_click'
            _last_index_click_time = now
    _index_click_was_above = index_above
    return pose

def _send_pose(pose, sent_commands):
    command = GESTURE_COMMANDS[pose]
    try:
        sent = send_hand_command(command)
    except Exception as exc:
        send_event_log(pose, command, status='error', message=str(exc))
        raise
    send_event_log(pose, sent, status='ok')
    sent_commands.append(sent)

def _recover_action(action, sent_commands):
    for pose in _as_tuple(RECOVERY_POSES.get(action)):
        _send_pose(pose, sent_commands)

def _activate_action(action, sent_commands):
    if action in _active_hand_actions:
        return
    _send_pose(action, sent_commands)
    _active_hand_actions.add(action)

def _recover_active_action(action, sent_commands):
    if action not in _active_hand_actions:
        return
    _recover_action(action, sent_commands)
    _active_hand_actions.discard(action)

def _recover_actions(actions, sent_commands):
    for action in actions:
        _recover_active_action(action, sent_commands)

def _recover_all_except(keep_actions, sent_commands):
    keep = set(keep_actions)
    for action in ACTIVE_ACTION_ORDER:
        if action not in keep:
            _recover_active_action(action, sent_commands)

def update_hand_trigger_from_encoders():
    """Send hand socket commands from encoder-derived stable poses and click edges."""
    if not _hand_trigger_enabled:
        return []

    positions = get_encoder_positions(use_calibration=True)
    sent_commands = []
    click_pose = _index_click_pose(positions)

    desired_pose = classify_encoder_pose(positions)
    if desired_pose in ('five_grasp', 'three_grasp_b'):
        _recover_all_except((desired_pose,), sent_commands)
        _activate_action(desired_pose, sent_commands)
        return sent_commands

    if click_pose is not None:
        _send_pose(click_pose, sent_commands)

    if desired_pose in ('index_point', 'index_press'):
        _recover_all_except((desired_pose,), sent_commands)
        _activate_action(desired_pose, sent_commands)
        return sent_commands

    _recover_actions(EXCLUSIVE_ACTIONS, sent_commands)

    thumb_active = desired_pose == 'thumb_in'
    if thumb_active:
        _activate_action('thumb_in', sent_commands)
    else:
        _recover_actions(('two_grasp_b', 'thumb_in'), sent_commands)
        return sent_commands

    thumb_flex = positions.get(2)
    two_grasp_active = 'thumb_in' in _active_hand_actions and _five_grasp_closed(thumb_flex)
    if two_grasp_active:
        _activate_action('two_grasp_b', sent_commands)
    else:
        _recover_active_action('two_grasp_b', sent_commands)

    return sent_commands

def get_encoder_readings():
    """Return raw PWM readings for debugging calibration quality."""
    readings = {}
    for i, pin in enumerate(ENCODER_PINS):
        readings[i + 1] = {
            'pin': pin,
            'pulse_width_us': _pulse_widths_us[pin],
            'period_us': _periods_us[pin],
            'duty': _duties[pin],
        }
    return readings

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Read PRISM RPi encoders and trigger hand socket commands.')
    parser.add_argument('--encoder-calibration', type=str, default=None,
                        help='calibration JSON path; omitted means newest configs/devices/encoder_pwm_calibration*.json')
    parser.add_argument('--hand-host', type=str, default=DEFAULT_HAND_HOST,
                        help='host IP running the hand socket server')
    parser.add_argument('--hand-port', type=int, default=DEFAULT_HAND_PORT,
                        help='host hand socket port')
    parser.add_argument('--hand-timeout-s', type=float, default=DEFAULT_HAND_TIMEOUT_S,
                        help='socket timeout for one-shot hand commands')
    parser.add_argument('--event-host', type=str, default=DEFAULT_EVENT_HOST,
                        help='PRISM collector host/IP for UDP event logging')
    parser.add_argument('--event-port', type=int, default=DEFAULT_EVENT_PORT,
                        help='PRISM collector UDP port for event logging; <=0 disables logging')
    parser.add_argument('--disable-event-log', action='store_true',
                        help='send hand commands without UDP event logging')
    parser.add_argument('--trigger-threshold', type=float, default=DEFAULT_TRIGGER_THRESHOLD,
                        help='normalized encoder threshold for hand triggers')
    parser.add_argument('--five-grasp-threshold', type=float, default=DEFAULT_FIVE_GRASP_THRESHOLD,
                        help='Enc2/3/4 normalized threshold for five_grasp')
    parser.add_argument('--thumb-in-threshold', type=float, default=DEFAULT_THUMB_IN_THRESHOLD,
                        help='Enc1 normalized value above this sends thumb_in; otherwise thumb_out on recovery')
    parser.add_argument('--index-press-min', type=float, default=DEFAULT_INDEX_PRESS_MIN,
                        help='minimum Enc5 normalized value for index_press when thumb/middle/ring are closed')
    parser.add_argument('--index-click-window-s', type=float, default=DEFAULT_INDEX_CLICK_WINDOW_S,
                        help='second Enc5 click within this window sends index_double_click')
    parser.add_argument('--disable-hand-trigger', action='store_true',
                        help='read encoders without sending hand socket commands')
    return parser.parse_args(argv)

# --- LED Control ---
leds = [LED(pin) for pin in LED_PINS]

def leds_on():
    """Turn all LEDs on constantly."""
    for led in leds:
        led.on()

def leds_off():
    """Turn all LEDs off."""
    for led in leds:
        led.off()

# --- Main ---
def main(argv=None):
    args = parse_args(argv)
    configure_hand_trigger(
        host=args.hand_host,
        port=args.hand_port,
        timeout_s=args.hand_timeout_s,
        threshold=args.trigger_threshold,
        five_grasp_threshold=args.five_grasp_threshold,
        thumb_in_threshold=args.thumb_in_threshold,
        index_press_min=args.index_press_min,
        index_click_window_s=args.index_click_window_s,
        enabled=not args.disable_hand_trigger,
        event_host=args.event_host,
        event_port=args.event_port,
        event_logging_enabled=not args.disable_event_log,
    )

    calibration = load_encoder_calibration(args.encoder_calibration)
    if calibration:
        loaded = ', '.join('Enc%d' % channel for channel in sorted(calibration))
        print("Loaded encoder calibration: %s (%s)" % (_encoder_calibration_path, loaded))
    else:
        print("No encoder calibration found, using fallback pulse-width mapping.")
    if _hand_trigger_enabled:
        print("Hand trigger target: %s:%d" % (_hand_host, _hand_port))
        if _event_logging_enabled and _event_host and _event_port > 0:
            print("Hand event log target: %s:%d" % (_event_host, _event_port))
        else:
            print("Hand event log disabled.")
        print("  Enc1 thumb swing: > %.1f%% -> thumb_in; recovery -> thumb_out" % (
            _thumb_in_threshold * 100.0,
        ))
        print("  Enc2 thumb flex, Enc3 ring, Enc4 middle, Enc5 index; flex threshold %.1f%%" % (
            _hand_trigger_threshold * 100.0,
        ))
        print("  Five grasp: Enc2/3/4 > %.1f%%; recovery -> five_open + thumb_out" % (
            _five_grasp_threshold * 100.0,
        ))
        print("  Two grasp B: thumb_in active and Enc2 > %.1f%%; Enc2 release sends two_open_b" % (
            _five_grasp_threshold * 100.0,
        ))
        print("  Stable poses: five_grasp, two_grasp_b, three_grasp_b, index_point, index_press, thumb_in")
        print("  Enc5 solo rising edge: single click; second edge within %.2fs: double click" % _index_click_window_s)
    else:
        print("Hand trigger disabled.")

    print("Setting up encoder PWM callbacks...")
    _setup_encoder_callbacks()

    print("Turning on all LEDs...")
    leds_on()

    print("Reading encoder angles. Press Ctrl+C to stop.\n")
    try:
        while True:
            angles = get_encoder_angles()
            try:
                sent_commands = update_hand_trigger_from_encoders()
                for sent in sent_commands:
                    print("\nHand command sent: %s" % sent)
            except Exception as exc:
                print("\nHand command send failed: %s" % exc)
            parts = [
                f"Enc{i}: {a:.1f}°" if a is not None else f"Enc{i}: ---"
                for i, a in angles.items()
            ]
            print("  |  ".join(parts), end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for cb in _callbacks:
            cb.cancel()
        pi.stop()
        leds_off()
        print("Done.")

if __name__ == "__main__":
    main()