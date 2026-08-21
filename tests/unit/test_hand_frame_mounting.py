import numpy as np

from prism.reconstruction.hand_frame import (
    DEFAULT_SPEC,
    apply_mount_correction,
    hand_pose_from_leds,
)


POINTS = {
    'red': np.array([0.0, 0.0, 0.0]),
    'blue': np.array([0.1, 0.0, 0.0]),
    'yellow': np.array([0.05, -0.05, 0.0]),
    'green': np.array([0.05, 0.05, 0.0]),
}


def test_reconstructed_led_pose_does_not_include_mount_correction():
    positive = dict(DEFAULT_SPEC)
    positive['mount_correction_deg'] = 45.0
    negative = dict(DEFAULT_SPEC)
    negative['mount_correction_deg'] = -45.0

    positive_pose = hand_pose_from_leds(POINTS, positive)
    negative_pose = hand_pose_from_leds(POINTS, negative)

    np.testing.assert_allclose(positive_pose[0], negative_pose[0], atol=1e-12)
    np.testing.assert_allclose(positive_pose[1], negative_pose[1], atol=1e-12)


def test_mount_correction_is_applied_only_for_robot_pose():
    spec = dict(DEFAULT_SPEC)
    spec['mount_correction_deg'] = -45.0
    led_rotation = hand_pose_from_leds(POINTS, spec)[0]
    robot_rotation = apply_mount_correction(led_rotation, spec)
    relative = led_rotation.T @ robot_rotation
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))

    assert abs(np.degrees(angle) - 45.0) < 1e-7


def test_origin_uses_blue_plus_configured_semantic_offsets():
    spec = dict(DEFAULT_SPEC)
    spec.update({
        'origin_forward_m': 0.005,
        'origin_right_m': 0.0,
        'origin_up_m': -0.03,
    })

    _, origin, _ = hand_pose_from_leds(POINTS, spec)

    # POINTS define forward=+X and right=+Y, so up=right x forward=-Z.
    np.testing.assert_allclose(origin, POINTS['blue'] + [0.005, 0.0, 0.03])