import importlib.util
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / 'tools' / 'replay_ur3_base_trajectory.py'
SPEC = importlib.util.spec_from_file_location('replay_ur3_base_trajectory', MODULE_PATH)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_time_scale_reduces_reported_speed():
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.02, 0.0, 0.0, 0.0, 0.0, np.deg2rad(4.0)],
    ], dtype=np.float64)

    normal = replay.trajectory_metrics(trajectory, time_scale=1.0)
    slow = replay.trajectory_metrics(trajectory, time_scale=2.0)

    assert slow['max_linear_step_m'] == normal['max_linear_step_m']
    assert slow['max_linear_speed_m_s'] == normal['max_linear_speed_m_s'] / 2.0
    assert slow['max_angular_speed_rad_s'] == normal['max_angular_speed_rad_s'] / 2.0


def test_recommended_time_scale_adds_margin_and_rounds_up():
    assert replay.recommended_time_scale(25, 1.288, 1.0) == 34.0


def test_metrics_reject_non_increasing_timestamps():
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)

    try:
        replay.trajectory_metrics(trajectory, time_scale=1.0)
    except ValueError as exc:
        assert 'strictly increasing' in str(exc)
    else:
        raise AssertionError('expected non-increasing timestamps to be rejected')


def test_resampling_preserves_endpoints_and_limits_steps():
    trajectory = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.04, 0.0, 0.0, 0.0, 0.0, np.deg2rad(20.0)],
    ], dtype=np.float64)

    output = replay.resample_trajectory(trajectory, time_scale=2.0, rate_hz=125.0)
    metrics = replay.trajectory_metrics(output, time_scale=2.0)

    np.testing.assert_allclose(output[0, 1:], trajectory[0, 1:], atol=1e-12)
    np.testing.assert_allclose(output[-1, 1:], trajectory[-1, 1:], atol=1e-12)
    assert len(output) == 51
    assert metrics['max_linear_step_m'] < 0.001
    assert np.rad2deg(metrics['max_angular_step_rad']) < 0.5


def test_pose_error_checks_position_and_orientation():
    target = np.array([0.4, -0.1, 0.3, 0.0, 0.0, np.deg2rad(30.0)])
    actual = np.array([0.403, -0.104, 0.3, 0.0, 0.0, np.deg2rad(32.0)])

    position_error, orientation_error = replay.pose_error(actual, target)

    assert abs(position_error - 0.005) < 1e-12
    assert abs(np.rad2deg(orientation_error) - 2.0) < 1e-10


def test_start_blend_reaches_target_in_small_pose_steps():
    before = np.array([0.4, -0.1, 0.3, 0.0, 0.0, np.deg2rad(28.0)])
    target = np.array([0.404, -0.1, 0.3, 0.0, 0.0, np.deg2rad(30.0)])

    poses = replay.interpolate_poses(before, target, duration_s=2.0, rate_hz=125.0)

    assert len(poses) == 250
    np.testing.assert_allclose(poses[-1], target, atol=1e-12)
    first_position_error, first_orientation_error = replay.pose_error(poses[0], before)
    assert first_position_error < 0.00002
    assert np.rad2deg(first_orientation_error) < 0.01


def test_ur3_plane_clearance_distinguishes_safe_and_low_branches():
    low = np.array([2.744376, 1.597628, -0.116646, -0.952920, 1.265575, -2.532567])
    high = np.array([1.896901, -2.045002, 2.335053, 0.276092, 2.527192, -0.143549])

    assert replay.ur3_plane_clearance(low, link_radius_m=0.04) < 0.0
    assert replay.ur3_plane_clearance(high, link_radius_m=0.04) > 0.10


def test_clearance_planner_prefers_higher_continuous_ik_branch():
    low = np.array([2.744376, 1.597628, -0.116646, -0.952920, 1.265575, -2.532567])
    high = np.array([1.896901, -2.045002, 2.335053, 0.276092, 2.527192, -0.143549])
    trajectory = np.array([
        [0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.1, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
    ])

    class Control:
        @staticmethod
        def getInverseKinematics(pose, qnear):
            return list(qnear)

    with patch.object(replay, 'initial_ik_candidates', return_value=[low, high]):
        path, clearances = replay.plan_clearance_aware_ik(
            Control(), trajectory, np.zeros(6), min_clearance_m=0.01,
            link_radius_m=0.04, max_source_joint_step_rad=0.5,
        )

    np.testing.assert_allclose(path[0], high)
    assert clearances[0] > clearances[1]


def test_ik_cache_round_trip_and_fingerprint_rejection(tmp_path):
    trajectory = np.array([
        [0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.1, 0.31, 0.0, 0.3, 0.0, 0.0, 0.0],
    ])
    joints = np.arange(12, dtype=np.float64).reshape(2, 6) * 0.01
    cache = tmp_path / 'ik_plan_cache.npz'

    replay.save_ik_cache(cache, trajectory, joints, '192.168.1.102', {
        'branch_clearances_m': [0.1],
    })
    loaded, error = replay.load_ik_cache(cache, trajectory, '192.168.1.102')

    assert error is None
    np.testing.assert_allclose(loaded[0], joints)
    assert loaded[1]['branch_clearances_m'] == [0.1]

    changed = trajectory.copy()
    changed[1, 1] += 0.001
    loaded, error = replay.load_ik_cache(cache, changed, '192.168.1.102')
    assert loaded is None
    assert error == 'trajectory fingerprint mismatch'


def test_cached_ik_validation_detects_wrong_joint_path():
    trajectory = np.array([
        [0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
        [0.1, 0.31, 0.0, 0.3, 0.0, 0.0, 0.0],
    ])
    joints = np.zeros((2, 6))

    class Control:
        @staticmethod
        def getInverseKinematics(pose, qnear):
            return np.asarray(qnear) + 0.1

    valid, joint_error = replay.validate_cached_ik(Control(), trajectory, joints)

    assert not valid
    assert joint_error > 0.2


def test_wrist_calibration_post_multiplies_local_tool_z():
    trajectory = np.array([
        [0.0, 0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
    ])
    corrected = replay.apply_local_wrist_rotation(trajectory, np.deg2rad(30.0))

    np.testing.assert_allclose(corrected[0, 1:4], trajectory[0, 1:4])
    assert abs(np.linalg.norm(corrected[0, 4:7]) - np.deg2rad(30.0)) < 1e-12
    np.testing.assert_allclose(corrected[0, 4:6], [0.0, 0.0], atol=1e-12)


def test_sdk_events_collapse_only_consecutive_duplicate_commands(tmp_path):
    path = tmp_path / 'sdk_commands.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['trial_time', 'action', 'command'])
        writer.writeheader()
        writer.writerows([
            {'trial_time': '0.1', 'action': 'a', 'command': '@ROG<11>&'},
            {'trial_time': '0.2', 'action': 'a', 'command': '@ROG<11>&'},
            {'trial_time': '0.3', 'action': 'b', 'command': '@ROG<12>&'},
            {'trial_time': '0.4', 'action': 'a', 'command': '@ROG<11>&'},
        ])

    events = replay.load_sdk_events(path)

    assert [(event[0], event[1]) for event in events] == [
        (0.1, '@ROG<11>&'),
        (0.3, '@ROG<12>&'),
        (0.4, '@ROG<11>&'),
    ]


def test_wrist_path_uses_equivalent_turn_to_avoid_limits():
    joints = np.zeros((3, 6))
    joints[:, 5] = np.deg2rad([204.0, 240.0, 271.0])

    centered, turns, margin = replay.center_periodic_joint_path(
        joints, joint_index=5, offset_rad=np.deg2rad(100.0)
    )

    assert turns == -1
    np.testing.assert_allclose(np.rad2deg(centered[:, 5]), [-56.0, -20.0, 11.0], atol=1e-10)
    assert np.rad2deg(margin) > 120.0