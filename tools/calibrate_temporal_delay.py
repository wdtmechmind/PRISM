#!/usr/bin/env python3
"""Deprecated temporal delay calibration entrypoint."""

import sys


def main():
    message = (
        'tools/calibrate_temporal_delay.py has been retired.\n'
        'PRISM now uses hardware-trigger synchronization, so temporal offset calibration is no longer supported.\n'
        'Use docs/HARDWARE_TRIGGER_CALIBRATION_GUIDE.md for the current workflow.'
    )
    print(message, file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
