import csv
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from prism.reconstruction import realtime_reconstruction as reconstruction


def test_two_view_triangulation_rejects_error_above_limit():
    observations = {0: (10.0, 20.0), 1: (30.0, 40.0)}
    bad_result = (np.array([0.1, 0.2, 0.8]), {0: 0.02, 1: 0.03})

    with patch.object(reconstruction, 'triangulate_multi_view', return_value=bad_result):
        point, errors = reconstruction.robust_triangulate(
            observations, cameras={}, max_norm_reproj_error=0.015
        )

    assert point is None
    assert errors is None


def test_first_measurement_does_not_create_prediction_velocity():
    state = reconstruction.make_track_state()
    args = SimpleNamespace(max_norm_reproj_error=0.015, max_predict_frames=6,
                           max_traj_points=100)
    observations = {name: {} for name in reconstruction.COLOR_ORDER}
    observations['green'] = {0: (10.0, 20.0), 1: (30.0, 40.0)}
    measured = np.array([0.1, -0.2, 0.8], dtype=np.float64)
    writer = csv.writer(io.StringIO())

    with patch.object(
        reconstruction, 'robust_triangulate', return_value=(measured, {0: 0.001, 1: 0.001})
    ):
        points, modes = reconstruction.advance_tracking(
            state, observations, cameras={}, args=args, dt=0.25,
            now=1.0, t0=0.0, writer=writer, paused=False,
        )

    assert modes['green'] == 'measured'
    np.testing.assert_allclose(points['green'], measured, atol=1e-7)

    missing = {name: {} for name in reconstruction.COLOR_ORDER}
    points, modes = reconstruction.advance_tracking(
        state, missing, cameras={}, args=args, dt=0.25,
        now=1.25, t0=0.0, writer=writer, paused=False,
    )

    assert modes['green'] == 'predicted'
    np.testing.assert_allclose(points['green'], measured, atol=1e-7)


def test_rigid_pose_drops_one_inconsistent_led():
    model = {
        'red': np.array([0.0, 0.0, 0.0]),
        'yellow': np.array([0.1, 0.0, 0.0]),
        'blue': np.array([0.0, 0.1, 0.0]),
        'green': np.array([0.1, 0.1, 0.02]),
    }
    translation = np.array([0.3, -0.2, 0.8])
    observed = {name: point + translation for name, point in model.items()}
    observed['green'] = observed['green'] + np.array([0.2, -0.1, 0.15])

    rotation, estimated_translation = reconstruction.estimate_pose_from_model(model, observed)

    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(estimated_translation, translation, atol=1e-8)


def test_rigid_pose_rejects_inconsistent_three_led_shape():
    model = {
        'red': np.array([0.0, 0.0, 0.0]),
        'yellow': np.array([0.1, 0.0, 0.0]),
        'blue': np.array([0.0, 0.1, 0.0]),
    }
    observed = {name: point.copy() for name, point in model.items()}
    observed['blue'] = observed['blue'] + np.array([0.1, 0.1, 0.1])

    assert reconstruction.estimate_pose_from_model(model, observed) is None