"""Pure intent-classification logic for RPi encoder-driven DexHand gestures.

This module is hardware-agnostic so rules can be unit tested without GPIO.
"""

from dataclasses import dataclass
from typing import Optional

OPEN = 'OPEN'
MID = 'MID'
CLOSED = 'CLOSED'
UNKNOWN = 'UNKNOWN'
IN = 'IN'
NORMAL_OR_OUT = 'NORMAL_OR_OUT'


@dataclass(frozen=True)
class EncoderIntentThresholds:
    open_threshold: float = 0.35
    closed_threshold: float = 0.60
    thumb_in_threshold: float = 0.60
    index_press_min: float = 0.20
    index_press_max: float = 0.60


@dataclass
class IndexClickDetectorState:
    was_index_closed: bool = False
    last_click_time: Optional[float] = None


def _classify_tristate(value, open_threshold, closed_threshold):
    if value is None:
        return UNKNOWN
    if value <= open_threshold:
        return OPEN
    if value >= closed_threshold:
        return CLOSED
    return MID


def classify_encoder_states(positions, thresholds):
    """Convert normalized encoder values to semantic intent states.

    positions keys are encoder channels 1..5.
    """
    thumb_swing_value = positions.get(1)
    thumb_flex_value = positions.get(2)
    ring_value = positions.get(3)
    middle_value = positions.get(4)
    index_value = positions.get(5)

    thumb_flex = _classify_tristate(
        thumb_flex_value,
        thresholds.open_threshold,
        thresholds.closed_threshold,
    )
    ring = _classify_tristate(ring_value, thresholds.open_threshold, thresholds.closed_threshold)
    middle = _classify_tristate(middle_value, thresholds.open_threshold, thresholds.closed_threshold)
    index = _classify_tristate(index_value, thresholds.open_threshold, thresholds.closed_threshold)

    thumb_swing = UNKNOWN
    if thumb_swing_value is not None:
        if thumb_swing_value >= thresholds.thumb_in_threshold:
            thumb_swing = IN
        else:
            thumb_swing = NORMAL_OR_OUT

    index_in_press_band = False
    if index_value is not None:
        index_in_press_band = thresholds.index_press_min <= index_value <= thresholds.index_press_max

    ready = all(positions.get(channel) is not None for channel in (1, 2, 3, 4, 5))
    return {
        'ready': ready,
        'thumb_swing': thumb_swing,
        'thumb_flex': thumb_flex,
        'ring': ring,
        'middle': middle,
        'index': index,
        'index_in_press_band': index_in_press_band,
    }


def classify_encoder_pose_from_states(states):
    """Classify stable-pose candidate from explicit rule table.

    Priority is encoded in rule order. Click events are handled separately.
    """
    if not states.get('ready', False):
        return None

    thumb_swing = states['thumb_swing']
    thumb_flex = states['thumb_flex']
    ring = states['ring']
    middle = states['middle']
    index = states['index']

    if (
        thumb_flex == CLOSED
        and ring == CLOSED
        and middle == CLOSED
        and states.get('index_in_press_band', False)
        and index != CLOSED
    ):
        return 'index_press'

    if thumb_flex == CLOSED and ring == CLOSED and middle == CLOSED and index == OPEN:
        return 'index_point'

    if (
        ring == CLOSED
        and middle == CLOSED
        and index == CLOSED
    ):
        return 'five_grasp'

    if (
        ring == OPEN
        and middle == CLOSED
        and index == CLOSED
    ):
        return 'three_grasp_b'

    if (
        ring == OPEN
        and middle == CLOSED
        and index == OPEN
    ):
        return 'two_grasp_b'

    if (
        thumb_swing == IN
        and thumb_flex == OPEN
        and ring == OPEN
        and middle == OPEN
        and index == OPEN
    ):
        return 'thumb_in'

    return None


def detect_index_click_event(states, detector_state, now_s, click_window_s):
    """Edge-triggered click detector with single/double click windowing."""
    if not states.get('ready', False):
        return None, detector_state

    click_context = (
        states.get('thumb_flex') == OPEN
        and states.get('ring') == OPEN
        and states.get('middle') == OPEN
    )
    index_closed = states.get('index') == CLOSED
    pose = None

    if index_closed and not detector_state.was_index_closed and click_context:
        if (
            detector_state.last_click_time is not None
            and now_s - detector_state.last_click_time <= click_window_s
        ):
            pose = 'index_double_click'
            detector_state.last_click_time = None
        else:
            pose = 'index_single_click'
            detector_state.last_click_time = now_s

    detector_state.was_index_closed = index_closed
    return pose, detector_state
