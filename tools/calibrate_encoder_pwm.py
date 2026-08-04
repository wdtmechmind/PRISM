#!/usr/bin/env python3
"""
Calibrate magnetic encoder PWM duty ranges on the PRISM Raspberry Pi IO board.

The existing online reader can show only a fixed pulse-width-to-angle mapping.
This tool records the observed duty-cycle min/max for each encoder while you
move the mechanism through its real mechanical range, then writes a JSON file
that can be used by the runtime angle conversion later.

Usage on the Raspberry Pi:
    sudo pigpiod
    python tools/calibrate_encoder_pwm.py --duration 30

During calibration, slowly move each joint through its full expected range.
Press Ctrl+C to finish early and save the best ranges collected so far.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(ROOT, 'configs', 'devices', 'encoder_pwm_calibration.json')
DEFAULT_PINS = [17, 27, 22, 5, 6]


def parse_csv_ints(text):
    try:
        values = [int(part.strip()) for part in text.split(',') if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from exc
    if not values:
        raise argparse.ArgumentTypeError('at least one integer is required')
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Calibrate PRISM magnetic encoder PWM duty-cycle ranges.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--pins', type=parse_csv_ints, default=DEFAULT_PINS,
                        help='GPIO BCM pins for encoder PWM inputs, ordered as Enc1..EncN')
    parser.add_argument('--channels', type=parse_csv_ints, default=None,
                        help='1-based encoder channels to include; omitted means all channels')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='calibration time in seconds; use 0 to run until Ctrl+C')
    parser.add_argument('--preview-interval', type=float, default=0.2,
                        help='seconds between live terminal updates')
    parser.add_argument('--min-period-us', type=float, default=100.0,
                        help='discard PWM cycles shorter than this period')
    parser.add_argument('--max-period-us', type=float, default=10000.0,
                        help='discard PWM cycles longer than this period')
    parser.add_argument('--min-span-duty', type=float, default=0.01,
                        help='warn when an observed duty span is smaller than this fraction')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='JSON path for the saved calibration')
    parser.add_argument('--no-save', action='store_true',
                        help='print results without writing JSON')
    return parser.parse_args(argv)


@dataclass
class EncoderState:
    channel: int
    pin: int
    high_us: float = 0.0
    period_us: float = 0.0
    duty: float = math.nan
    min_high_us: float = math.inf
    max_high_us: float = -math.inf
    min_period_us: float = math.inf
    max_period_us: float = -math.inf
    min_duty: float = math.inf
    max_duty: float = -math.inf
    samples: int = 0
    rejected_cycles: int = 0
    last_rise_tick: int = None
    last_update_monotonic: float = 0.0
    lock: Lock = field(default_factory=Lock)

    def observe_rise(self, tick, pigpio_module):
        with self.lock:
            if self.last_rise_tick is not None:
                self.period_us = float(pigpio_module.tickDiff(self.last_rise_tick, tick))
            self.last_rise_tick = tick

    def observe_fall(self, tick, pigpio_module, min_period_us, max_period_us):
        with self.lock:
            if self.last_rise_tick is None or self.period_us <= 0.0:
                return

            high_us = float(pigpio_module.tickDiff(self.last_rise_tick, tick))
            period_us = self.period_us
            if high_us <= 0.0 or high_us > period_us:
                self.rejected_cycles += 1
                return
            if period_us < min_period_us or period_us > max_period_us:
                self.rejected_cycles += 1
                return

            duty = high_us / period_us
            if duty <= 0.0 or duty >= 1.0:
                self.rejected_cycles += 1
                return

            self.high_us = high_us
            self.duty = duty
            self.min_high_us = min(self.min_high_us, high_us)
            self.max_high_us = max(self.max_high_us, high_us)
            self.min_period_us = min(self.min_period_us, period_us)
            self.max_period_us = max(self.max_period_us, period_us)
            self.min_duty = min(self.min_duty, duty)
            self.max_duty = max(self.max_duty, duty)
            self.samples += 1
            self.last_update_monotonic = time.monotonic()

    def snapshot(self):
        with self.lock:
            has_samples = self.samples > 0
            min_duty = self.min_duty if has_samples else None
            max_duty = self.max_duty if has_samples else None
            return {
                'channel': self.channel,
                'name': 'Enc%d' % self.channel,
                'pin': self.pin,
                'samples': self.samples,
                'rejected_cycles': self.rejected_cycles,
                'latest_high_us': self.high_us if has_samples else None,
                'latest_period_us': self.period_us if has_samples else None,
                'latest_duty': self.duty if has_samples else None,
                'min_high_us': self.min_high_us if has_samples else None,
                'max_high_us': self.max_high_us if has_samples else None,
                'min_period_us': self.min_period_us if has_samples else None,
                'max_period_us': self.max_period_us if has_samples else None,
                'min_duty': min_duty,
                'max_duty': max_duty,
                'span_duty': (max_duty - min_duty) if has_samples else None,
                'center_duty': ((min_duty + max_duty) / 2.0) if has_samples else None,
                'age_s': (time.monotonic() - self.last_update_monotonic) if has_samples else None,
            }


class PwmCalibrator:
    def __init__(self, args, pigpio_module):
        self.args = args
        self.pigpio = pigpio_module
        selected_channels = set(args.channels) if args.channels else set(range(1, len(args.pins) + 1))
        self.states = [
            EncoderState(channel=index + 1, pin=pin)
            for index, pin in enumerate(args.pins)
            if index + 1 in selected_channels
        ]
        if not self.states:
            raise ValueError('no channels selected')
        unknown_channels = selected_channels - {state.channel for state in self.states}
        if unknown_channels:
            raise ValueError('selected channels exceed pin list: %s' % sorted(unknown_channels))
        self.pi = None
        self.callbacks = []

    def __enter__(self):
        self.pi = self.pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError('pigpio daemon not running. Run: sudo pigpiod')

        for state in self.states:
            self.pi.set_mode(state.pin, self.pigpio.INPUT)
            callback = self.pi.callback(state.pin, self.pigpio.EITHER_EDGE, self._make_callback(state))
            self.callbacks.append(callback)
        return self

    def __exit__(self, exc_type, exc, tb):
        for callback in self.callbacks:
            callback.cancel()
        if self.pi is not None:
            self.pi.stop()

    def _make_callback(self, state):
        def callback(gpio, level, tick):
            if gpio != state.pin:
                return
            if level == 1:
                state.observe_rise(tick, self.pigpio)
            elif level == 0:
                state.observe_fall(
                    tick,
                    self.pigpio,
                    self.args.min_period_us,
                    self.args.max_period_us,
                )

        return callback

    def snapshots(self):
        return [state.snapshot() for state in self.states]


def format_percent(value):
    if value is None or not math.isfinite(value):
        return '---'
    return '%6.2f%%' % (value * 100.0)


def print_live_line(snapshots):
    parts = []
    for item in snapshots:
        parts.append(
            '%s duty=%s range=%s..%s span=%s n=%d' % (
                item['name'],
                format_percent(item['latest_duty']),
                format_percent(item['min_duty']),
                format_percent(item['max_duty']),
                format_percent(item['span_duty']),
                item['samples'],
            )
        )
    line = ' | '.join(parts)
    print(line[:220].ljust(220), end='\r', flush=True)


def build_output(args, started_at, finished_at, snapshots):
    encoders = {}
    for item in snapshots:
        encoders[item['name']] = item
    return {
        'schema': 'prism.encoder_pwm_calibration.v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'started_monotonic_s': started_at,
        'finished_monotonic_s': finished_at,
        'duration_s': finished_at - started_at,
        'pins': args.pins,
        'period_filter_us': {
            'min': args.min_period_us,
            'max': args.max_period_us,
        },
        'angle_mapping_hint': {
            'description': 'For each encoder, clamp duty to [min_duty, max_duty], then normalize to the observed mechanical range.',
            'normalized_position': '(duty - min_duty) / (max_duty - min_duty)',
        },
        'encoders': encoders,
    }


def print_summary(snapshots, min_span_duty):
    print('\nCalibration summary:')
    for item in snapshots:
        span = item['span_duty']
        status = 'OK'
        if item['samples'] == 0:
            status = 'NO_SIGNAL'
        elif span is None or span < min_span_duty:
            status = 'SMALL_SPAN'
        print(
            '  {name} pin={pin:<2d} samples={samples:<6d} duty={lo}..{hi} span={span} status={status}'.format(
                name=item['name'],
                pin=item['pin'],
                samples=item['samples'],
                lo=format_percent(item['min_duty']),
                hi=format_percent(item['max_duty']),
                span=format_percent(item['span_duty']),
                status=status,
            )
        )
    print('')
    print('Tip: SMALL_SPAN means that channel did not move enough during this run,')
    print('     or the magnet initial angle/mechanical linkage keeps it in a narrow PWM band.')


def load_pigpio():
    try:
        import pigpio
    except ImportError as exc:
        raise RuntimeError('pigpio Python package is not installed on this interpreter') from exc
    return pigpio


def main(argv=None):
    args = parse_args(argv)
    pigpio_module = load_pigpio()

    print('PRISM encoder PWM calibration')
    print('Pins:', ', '.join('Enc%d=GPIO%d' % (index + 1, pin) for index, pin in enumerate(args.pins)))
    print('Move selected joints through their full range. Press Ctrl+C to finish early.')
    if args.duration > 0:
        print('Duration: %.1f s' % args.duration)
    else:
        print('Duration: until Ctrl+C')
    print('')

    started_at = time.monotonic()
    snapshots = []
    try:
        with PwmCalibrator(args, pigpio_module) as calibrator:
            while True:
                now = time.monotonic()
                snapshots = calibrator.snapshots()
                print_live_line(snapshots)
                if args.duration > 0 and now - started_at >= args.duration:
                    break
                time.sleep(max(0.02, args.preview_interval))
    except KeyboardInterrupt:
        print('\nStopped by user.')
    finally:
        finished_at = time.monotonic()

    if not snapshots:
        print('No snapshots collected.', file=sys.stderr)
        return 2

    print_summary(snapshots, args.min_span_duty)
    result = build_output(args, started_at, finished_at, snapshots)

    if args.no_save:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print('Saved calibration:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())