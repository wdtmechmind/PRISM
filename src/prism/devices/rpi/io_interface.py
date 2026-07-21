"""
RPi GPIO interface for 5-channel magnetic encoder PWM reading and LED control.

Encoder PWM inputs:  GPIO 17, 27, 22, 5, 6  (3.3V PWM signals)
LED outputs:         GPIO 23, 24, 25, 12, 16

Uses gpiozero for GPIO control.
"""

import time
import threading
from gpiozero import LED, AngularServo
from gpiozero.pins.rpigpio import RPiGPIOFactory
from gpiozero import Device
import pigpio

# --- Pin Definitions ---
ENCODER_PINS = [17, 27, 22, 5, 6]   # PWM input pins for encoders 1-5
LED_PINS     = [23, 24, 25, 12, 16]  # Output pins for LEDs 1-5

# --- PWM Reception via pigpio ---
# gpiozero does not natively read PWM duty cycle/frequency.
# We use pigpio callbacks to measure pulse widths for encoder angle decoding.

pi = pigpio.pi()
if not pi.connected:
    raise RuntimeError("pigpio daemon not running. Run: sudo pigpiod")

# Storage for latest pulse width (microseconds) per encoder pin
_pulse_widths_us = {pin: 0 for pin in ENCODER_PINS}
_callbacks = []

def _make_pwm_callback(pin):
    """Create a pigpio callback to measure PWM high pulse width on a pin."""
    _last_tick = [None]

    def _cb(gpio, level, tick):
        if level == 1:
            _last_tick[0] = tick
        elif level == 0 and _last_tick[0] is not None:
            pulse_width = pigpio.tickDiff(_last_tick[0], tick)
            _pulse_widths_us[gpio] = pulse_width

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

def get_encoder_angles():
    """Return dict of {encoder_index: angle_degrees} for encoders 1-5."""
    angles = {}
    for i, pin in enumerate(ENCODER_PINS):
        pw = _pulse_widths_us[pin]
        angles[i + 1] = pulse_width_to_angle(pw) if pw > 0 else None
    return angles

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
def main():
    print("Setting up encoder PWM callbacks...")
    _setup_encoder_callbacks()

    print("Turning on all LEDs...")
    leds_on()

    print("Reading encoder angles. Press Ctrl+C to stop.\n")
    try:
        while True:
            angles = get_encoder_angles()
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