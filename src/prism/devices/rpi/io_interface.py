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
from prism.devices.rpi.gesture_intent import (
    EncoderIntentThresholds,
    IN,
    IndexClickDetectorState,
    classify_encoder_states,
    classify_encoder_pose_from_states,
    detect_index_click_event,
)

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
DEFAULT_OPEN_THRESHOLD = float(os.environ.get('PRISM_HAND_OPEN_THRESHOLD', '0.35'))
DEFAULT_CLOSED_THRESHOLD = float(os.environ.get('PRISM_HAND_CLOSED_THRESHOLD', '0.60'))
DEFAULT_THUMB_IN_THRESHOLD = float(os.environ.get('PRISM_THUMB_IN_THRESHOLD', '0.60'))
DEFAULT_INDEX_PRESS_MIN = float(os.environ.get('PRISM_INDEX_PRESS_MIN', '0.2'))
DEFAULT_INDEX_PRESS_MAX = float(os.environ.get('PRISM_INDEX_PRESS_MAX', '0.6'))
DEFAULT_STABLE_TIME_S = float(os.environ.get('PRISM_HAND_STABLE_TIME_S', '0.12'))
DEFAULT_COOLDOWN_S = float(os.environ.get('PRISM_HAND_COOLDOWN_S', '0.2'))
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
    'five_grasp': 'five_open',
    'two_grasp_b': 'two_open_b',
    'three_grasp_b': 'three_open_b',
    'index_point': 'five_open',
    'index_press': 'five_open',
    'thumb_in': 'thumb_out',
}

EXCLUSIVE_ACTIONS = ('index_press', 'index_point', 'five_grasp', 'three_grasp_b', 'two_grasp_b')
ACTIVE_ACTION_ORDER = ('two_grasp_b', 'five_grasp', 'three_grasp_b', 'index_point', 'index_press')

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
_open_threshold = DEFAULT_OPEN_THRESHOLD
_closed_threshold = DEFAULT_CLOSED_THRESHOLD
_thumb_in_threshold = DEFAULT_THUMB_IN_THRESHOLD
_index_press_min = DEFAULT_INDEX_PRESS_MIN
_index_press_max = DEFAULT_INDEX_PRESS_MAX
_stable_time_s = DEFAULT_STABLE_TIME_S
_cooldown_s = DEFAULT_COOLDOWN_S
_index_click_window_s = DEFAULT_INDEX_CLICK_WINDOW_S
_hand_trigger_enabled = True
_active_hand_actions = set()
_index_click_state = IndexClickDetectorState()
_candidate_pose = None
_candidate_pose_since_s = None
_stable_pose = None
_last_sent_action_times_s = {}
_gesture_debug_context = {}
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
                           index_press_min=None, index_press_max=None,
                           open_threshold=None, closed_threshold=None,
                           stable_time_s=None, cooldown_s=None,
                           index_click_window_s=None, enabled=True,
                           event_host=None, event_port=None, event_logging_enabled=True):
    """Configure edge-triggered hand socket commands from encoder state."""
    global _hand_host, _hand_port, _hand_timeout_s, _hand_trigger_threshold, _five_grasp_threshold
    global _thumb_in_threshold, _index_press_min, _index_press_max, _index_click_window_s
    global _open_threshold, _closed_threshold, _stable_time_s, _cooldown_s
    global _hand_trigger_enabled
    global _event_host, _event_port, _event_logging_enabled
    global _index_click_state, _candidate_pose, _candidate_pose_since_s, _stable_pose
    global _last_sent_action_times_s
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
    if open_threshold is not None:
        _open_threshold = float(open_threshold)
    if closed_threshold is not None:
        _closed_threshold = float(closed_threshold)
    if threshold is not None and open_threshold is None and closed_threshold is None:
        legacy = float(threshold)
        _open_threshold = legacy
        _closed_threshold = legacy
    if five_grasp_threshold is not None and closed_threshold is None:
        _closed_threshold = float(five_grasp_threshold)
    if thumb_in_threshold is not None:
        _thumb_in_threshold = float(thumb_in_threshold)
    if index_press_min is not None:
        _index_press_min = float(index_press_min)
    if index_press_max is not None:
        _index_press_max = float(index_press_max)
    if stable_time_s is not None:
        _stable_time_s = max(0.0, float(stable_time_s))
    if cooldown_s is not None:
        _cooldown_s = max(0.0, float(cooldown_s))
    if index_click_window_s is not None:
        _index_click_window_s = float(index_click_window_s)
    if event_host is not None:
        _event_host = event_host
    if event_port is not None:
        _event_port = int(event_port)
    _event_logging_enabled = bool(event_logging_enabled)
    _hand_trigger_enabled = bool(enabled)
    _index_click_state = IndexClickDetectorState()
    _candidate_pose = None
    _candidate_pose_since_s = None
    _stable_pose = None
    _last_sent_action_times_s = {}
    _active_hand_actions.clear()

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

def send_event_log(action, command, status='ok', message='', debug=None):
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
    if debug:
        payload['encoder_states'] = debug.get('encoder_states')
        payload['candidate_pose'] = debug.get('candidate_pose')
        payload['stable_pose'] = debug.get('stable_pose')
        payload['click_event'] = debug.get('click_event')
        payload['active_actions'] = debug.get('active_actions')
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(data, (_event_host, int(_event_port)))
    return True

def _positions_ready(positions):
    return all(positions.get(channel) is not None for channel in range(1, 6))

def _build_thresholds():
    return EncoderIntentThresholds(
        open_threshold=_open_threshold,
        closed_threshold=_closed_threshold,
        thumb_in_threshold=_thumb_in_threshold,
        index_press_min=_index_press_min,
        index_press_max=_index_press_max,
    )

def _as_tuple(value):
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    return (value,)

def classify_encoder_pose(positions):
    """Map Enc1..Enc5 calibrated positions to one stable candidate pose."""
    states = classify_encoder_states(positions, _build_thresholds())
    return classify_encoder_pose_from_states(states)

def _index_click_pose(positions):
    global _index_click_state
    states = classify_encoder_states(positions, _build_thresholds())
    pose, _index_click_state = detect_index_click_event(
        states,
        _index_click_state,
        time.monotonic(),
        _index_click_window_s,
    )
    return pose

def _in_action_cooldown(action, now_s):
    if _cooldown_s <= 0.0:
        return False
    last = _last_sent_action_times_s.get(action)
    return last is not None and now_s - last < _cooldown_s

def _send_pose(pose, sent_commands, now_s=None, apply_cooldown=False):
    if now_s is None:
        now_s = time.monotonic()
    if apply_cooldown and _in_action_cooldown(pose, now_s):
        return False
    command = GESTURE_COMMANDS[pose]
    try:
        sent = send_hand_command(command)
    except Exception as exc:
        send_event_log(pose, command, status='error', message=str(exc), debug=_gesture_debug_context)
        raise
    send_event_log(pose, sent, status='ok', debug=_gesture_debug_context)
    if apply_cooldown:
        _last_sent_action_times_s[pose] = now_s
    sent_commands.append(sent)
    return True

def _recover_action(action, sent_commands):
    for pose in _as_tuple(RECOVERY_POSES.get(action)):
        _send_pose(pose, sent_commands)

def _activate_action(action, sent_commands, now_s=None):
    if action in _active_hand_actions:
        return
    if _send_pose(action, sent_commands, now_s=now_s, apply_cooldown=True):
        _active_hand_actions.add(action)

def _recover_active_action(action, sent_commands, states=None):
    if action not in _active_hand_actions:
        return
    if action == 'thumb_in' and states is not None:
        # Thumb-out should only happen when Enc1 actually leaves IN.
        if states.get('thumb_swing') == IN:
            return
    _recover_action(action, sent_commands)
    _active_hand_actions.discard(action)

def _recover_actions(actions, sent_commands, states=None):
    for action in actions:
        _recover_active_action(action, sent_commands, states=states)

def _recover_all_except(keep_actions, sent_commands, states=None):
    keep = set(keep_actions)
    for action in ACTIVE_ACTION_ORDER:
        if action not in keep:
            _recover_active_action(action, sent_commands, states=states)

def _update_stable_pose(candidate_pose, now_s):
    global _candidate_pose, _candidate_pose_since_s, _stable_pose

    if candidate_pose != _candidate_pose:
        _candidate_pose = candidate_pose
        _candidate_pose_since_s = now_s

    if _candidate_pose_since_s is None:
        _candidate_pose_since_s = now_s

    held_long_enough = (now_s - _candidate_pose_since_s) >= _stable_time_s
    if held_long_enough:
        _stable_pose = candidate_pose
    return _stable_pose

def update_hand_trigger_from_encoders():
    """Send hand socket commands from encoder-derived stable poses and click edges."""
    global _gesture_debug_context
    if not _hand_trigger_enabled:
        return []

    now_s = time.monotonic()
    positions = get_encoder_positions(use_calibration=True)
    thresholds = _build_thresholds()
    states = classify_encoder_states(positions, thresholds)
    candidate_pose = classify_encoder_pose_from_states(states)
    stable_pose = _update_stable_pose(candidate_pose, now_s)

    sent_commands = []
    click_pose, _ = detect_index_click_event(
        states,
        _index_click_state,
        now_s,
        _index_click_window_s,
    )

    _gesture_debug_context = {
        'encoder_states': states,
        'candidate_pose': candidate_pose,
        'stable_pose': stable_pose,
        'click_event': click_pose,
        'active_actions': sorted(_active_hand_actions),
    }

    if click_pose is not None:
        _send_pose(click_pose, sent_commands, now_s=now_s, apply_cooldown=True)

    if stable_pose in EXCLUSIVE_ACTIONS:
        _recover_all_except((stable_pose,), sent_commands, states=states)
        _activate_action(stable_pose, sent_commands, now_s=now_s)
    else:
        _recover_actions(EXCLUSIVE_ACTIONS, sent_commands, states=states)

    thumb_mode_active = states.get('thumb_swing') == IN
    if thumb_mode_active:
        _activate_action('thumb_in', sent_commands, now_s=now_s)
    else:
        _recover_active_action('thumb_in', sent_commands, states=states)

    _gesture_debug_context['active_actions'] = sorted(_active_hand_actions)

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
    parser.add_argument('--trigger-threshold', type=float, default=None,
                        help='legacy single-threshold mode; if open/closed thresholds are unset, both use this value')
    parser.add_argument('--five-grasp-threshold', type=float, default=None,
                        help='legacy alias for closed-threshold when closed-threshold is unset')
    parser.add_argument('--open-threshold', type=float, default=None,
                        help='normalized value <= this is OPEN')
    parser.add_argument('--closed-threshold', type=float, default=None,
                        help='normalized value >= this is CLOSED')
    parser.add_argument('--thumb-in-threshold', type=float, default=DEFAULT_THUMB_IN_THRESHOLD,
                        help='Enc1 normalized value >= this is thumb swing IN (otherwise NORMAL_OR_OUT)')
    parser.add_argument('--index-press-min', type=float, default=DEFAULT_INDEX_PRESS_MIN,
                        help='minimum Enc5 normalized value for index_press band')
    parser.add_argument('--index-press-max', type=float, default=DEFAULT_INDEX_PRESS_MAX,
                        help='maximum Enc5 normalized value for index_press band')
    parser.add_argument('--stable-time-s', type=float, default=DEFAULT_STABLE_TIME_S,
                        help='candidate pose must hold this long before activation')
    parser.add_argument('--cooldown-s', type=float, default=DEFAULT_COOLDOWN_S,
                        help='minimum interval between repeated sends of the same action')
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

def _format_intent_debug_text():
    states = _gesture_debug_context.get('encoder_states') or {}
    if not states:
        return "states: n/a"

    thumb_swing = states.get('thumb_swing', '?')
    thumb_flex = states.get('thumb_flex', '?')
    ring = states.get('ring', '?')
    middle = states.get('middle', '?')
    index = states.get('index', '?')
    press_band = 'Y' if states.get('index_in_press_band') else 'N'
    candidate = _gesture_debug_context.get('candidate_pose') or 'idle'
    stable = _gesture_debug_context.get('stable_pose') or 'idle'
    click = _gesture_debug_context.get('click_event') or '-'
    active = ','.join(_gesture_debug_context.get('active_actions') or []) or '-'

    return (
        "S[1:%s 2:%s 3:%s 4:%s 5:%s PB:%s] C:%s ST:%s CLK:%s A:%s"
        % (thumb_swing, thumb_flex, ring, middle, index, press_band, candidate, stable, click, active)
    )

# --- Main ---
def main(argv=None):
    args = parse_args(argv)
    configure_hand_trigger(
        host=args.hand_host,
        port=args.hand_port,
        timeout_s=args.hand_timeout_s,
        threshold=args.trigger_threshold,
        five_grasp_threshold=args.five_grasp_threshold,
        open_threshold=args.open_threshold,
        closed_threshold=args.closed_threshold,
        thumb_in_threshold=args.thumb_in_threshold,
        index_press_min=args.index_press_min,
        index_press_max=args.index_press_max,
        stable_time_s=args.stable_time_s,
        cooldown_s=args.cooldown_s,
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
        print("  Thresholds: OPEN <= %.1f%%, CLOSED >= %.1f%%" % (
            _open_threshold * 100.0,
            _closed_threshold * 100.0,
        ))
        print("  Enc1 thumb swing: >= %.1f%% -> IN, else NORMAL_OR_OUT" % (
            _thumb_in_threshold * 100.0,
        ))
        print("  Index press band: %.1f%% .. %.1f%% (while Enc2/3/4 CLOSED)" % (
            _index_press_min * 100.0,
            _index_press_max * 100.0,
        ))
        print("  Stable confirmation: %.3fs, action cooldown: %.3fs" % (
            _stable_time_s,
            _cooldown_s,
        ))
        print("  Stable poses priority: index_press > index_point > five_grasp > three_grasp_b > two_grasp_b > thumb_in")
        print("  Enc5 solo rising edge: single click; second edge within %.2fs: double click" % _index_click_window_s)
    else:
        print("Hand trigger disabled.")

    print("Setting up encoder PWM callbacks...")
    _setup_encoder_callbacks()

    print("Turning on all LEDs...")
    leds_on()

    print("Reading encoder angles. Press Ctrl+C to stop.\n")
    print("Debug fields: S=encoder states, C=candidate pose, ST=stable pose, CLK=click event, A=active actions")
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
            intent_debug = _format_intent_debug_text()
            print("  |  ".join(parts) + "  ||  " + intent_debug, end="\r")
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