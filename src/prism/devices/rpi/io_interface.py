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
DEFAULT_TRIGGER_THRESHOLD = float(os.environ.get('PRISM_HAND_TRIGGER_THRESHOLD', '0.5'))
FIVE_GRASP_CHANNELS = (3, 4, 5)
THREE_GRASP_B_CHANNELS = (4, 5)
INDEX_CLICK_CHANNEL = 5
FIVE_GRASP_COMMAND = '@ROG<0>&'
FIVE_OPEN_COMMAND = '@ROG<1>&'
THREE_GRASP_B_COMMAND = '@ROG<8>&'
THREE_OPEN_B_COMMAND = '@ROG<9>&'
INDEX_SINGLE_CLICK_COMMAND = '@ROG<15>&'

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
_hand_trigger_threshold = DEFAULT_TRIGGER_THRESHOLD
_hand_trigger_enabled = True
_last_five_command = FIVE_OPEN_COMMAND
_last_three_b_command = THREE_OPEN_B_COMMAND
_index_click_was_above = False
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

def configure_hand_trigger(host=None, port=None, timeout_s=None, threshold=None, enabled=True):
    """Configure edge-triggered hand socket commands from encoder state."""
    global _hand_host, _hand_port, _hand_timeout_s, _hand_trigger_threshold, _hand_trigger_enabled
    if host is not None:
        _hand_host = host
    if port is not None:
        _hand_port = int(port)
    if timeout_s is not None:
        _hand_timeout_s = float(timeout_s)
    if threshold is not None:
        _hand_trigger_threshold = float(threshold)
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

def _all_channels_above(positions, channels):
    values = [positions.get(channel) for channel in channels]
    if any(value is None for value in values):
        return None
    return all(value > _hand_trigger_threshold for value in values)

def update_hand_trigger_from_encoders():
    """Send hand socket commands only on configured encoder threshold edges."""
    global _last_five_command, _last_three_b_command, _index_click_was_above
    if not _hand_trigger_enabled:
        return []

    positions = get_encoder_positions(use_calibration=True)
    sent_commands = []

    five_above = _all_channels_above(positions, FIVE_GRASP_CHANNELS)
    five_changed = False
    if five_above is not None:
        desired_command = FIVE_GRASP_COMMAND if five_above else FIVE_OPEN_COMMAND
        if desired_command != _last_five_command:
            sent = send_hand_command(desired_command)
            _last_five_command = desired_command
            sent_commands.append(sent)
            five_changed = True

    three_b_above = _all_channels_above(positions, THREE_GRASP_B_CHANNELS)
    three_b_changed = False
    if three_b_above is not None:
        desired_command = THREE_GRASP_B_COMMAND if three_b_above else THREE_OPEN_B_COMMAND
        if desired_command != _last_three_b_command:
            sent = send_hand_command(desired_command)
            _last_three_b_command = desired_command
            sent_commands.append(sent)
            three_b_changed = True

    index_position = positions.get(INDEX_CLICK_CHANNEL)
    if index_position is not None:
        index_above = index_position > _hand_trigger_threshold
        if index_above and not _index_click_was_above and not five_changed and not three_b_changed:
            sent = send_hand_command(INDEX_SINGLE_CLICK_COMMAND)
            sent_commands.append(sent)
        _index_click_was_above = index_above

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
    parser.add_argument('--trigger-threshold', type=float, default=DEFAULT_TRIGGER_THRESHOLD,
                        help='normalized encoder threshold for hand triggers')
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
        enabled=not args.disable_hand_trigger,
    )

    calibration = load_encoder_calibration(args.encoder_calibration)
    if calibration:
        loaded = ', '.join('Enc%d' % channel for channel in sorted(calibration))
        print("Loaded encoder calibration: %s (%s)" % (_encoder_calibration_path, loaded))
    else:
        print("No encoder calibration found, using fallback pulse-width mapping.")
    if _hand_trigger_enabled:
        print("Hand trigger target: %s:%d" % (_hand_host, _hand_port))
        print("  Enc3/4/5 > %.1f%% -> five_grasp, otherwise five_open" % (
            _hand_trigger_threshold * 100.0,
        ))
        print("  Enc4/5   > %.1f%% -> three_grasp_b, otherwise three_open_b" % (_hand_trigger_threshold * 100.0))
        print("  Enc5 rising above %.1f%% -> index_single_click" % (_hand_trigger_threshold * 100.0))
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