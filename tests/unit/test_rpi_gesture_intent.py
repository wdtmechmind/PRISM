import unittest

from prism.devices.rpi.gesture_intent import (
    CLOSED,
    IN,
    MID,
    NORMAL_OR_OUT,
    OPEN,
    EncoderIntentThresholds,
    IndexClickDetectorState,
    classify_encoder_pose_from_states,
    classify_encoder_states,
    detect_index_click_event,
)


class GestureIntentTests(unittest.TestCase):
    def setUp(self):
        self.t = EncoderIntentThresholds(
            open_threshold=0.35,
            closed_threshold=0.60,
            thumb_in_threshold=0.60,
            index_press_min=0.20,
            index_press_max=0.60,
        )

    def test_index_press_and_index_point_do_not_overlap(self):
        press_positions = {1: 0.30, 2: 0.80, 3: 0.80, 4: 0.80, 5: 0.45}
        point_positions = {1: 0.30, 2: 0.80, 3: 0.80, 4: 0.80, 5: 0.10}

        press_states = classify_encoder_states(press_positions, self.t)
        point_states = classify_encoder_states(point_positions, self.t)

        self.assertEqual(classify_encoder_pose_from_states(press_states), 'index_press')
        self.assertEqual(classify_encoder_pose_from_states(point_states), 'index_point')

    def test_five_grasp_not_cover_press_or_point(self):
        five_positions = {1: 0.20, 2: 0.80, 3: 0.80, 4: 0.80, 5: 0.80}
        press_positions = {1: 0.20, 2: 0.80, 3: 0.80, 4: 0.80, 5: 0.45}

        five_states = classify_encoder_states(five_positions, self.t)
        press_states = classify_encoder_states(press_positions, self.t)

        self.assertEqual(classify_encoder_pose_from_states(five_states), 'five_grasp')
        self.assertEqual(classify_encoder_pose_from_states(press_states), 'index_press')

    def test_three_grasp_b_distinct_from_five_grasp(self):
        five_positions = {1: 0.80, 2: 0.10, 3: 0.80, 4: 0.80, 5: 0.80}
        three_positions = {1: 0.80, 2: 0.10, 3: 0.10, 4: 0.80, 5: 0.80}

        self.assertEqual(
            classify_encoder_pose_from_states(classify_encoder_states(five_positions, self.t)),
            'five_grasp',
        )
        self.assertEqual(
            classify_encoder_pose_from_states(classify_encoder_states(three_positions, self.t)),
            'three_grasp_b',
        )

    def test_two_grasp_b_is_direct_state_based(self):
        two_positions = {1: 0.20, 2: 0.10, 3: 0.10, 4: 0.80, 5: 0.10}
        states = classify_encoder_states(two_positions, self.t)
        self.assertEqual(states['thumb_swing'], NORMAL_OR_OUT)
        self.assertEqual(classify_encoder_pose_from_states(states), 'two_grasp_b')

    def test_two_grasp_b_can_overlay_thumb_in_mode(self):
        thumb_in_two_positions = {1: 0.80, 2: 0.10, 3: 0.10, 4: 0.80, 5: 0.10}
        states = classify_encoder_states(thumb_in_two_positions, self.t)
        self.assertEqual(states['thumb_swing'], IN)
        self.assertEqual(classify_encoder_pose_from_states(states), 'two_grasp_b')

    def test_thumb_in_requires_other_fingers_open(self):
        thumb_in_positions = {1: 0.80, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10}
        blocked_positions = {1: 0.80, 2: 0.10, 3: 0.10, 4: 0.80, 5: 0.10}

        self.assertEqual(
            classify_encoder_pose_from_states(classify_encoder_states(thumb_in_positions, self.t)),
            'thumb_in',
        )
        self.assertEqual(
            classify_encoder_pose_from_states(classify_encoder_states(blocked_positions, self.t)),
            'two_grasp_b',
        )

    def test_thumb_in_can_overlay_three_and_five_grasp(self):
        thumb_five_positions = {1: 0.80, 2: 0.10, 3: 0.80, 4: 0.80, 5: 0.80}
        thumb_three_positions = {1: 0.80, 2: 0.10, 3: 0.10, 4: 0.80, 5: 0.80}

        self.assertEqual(
            classify_encoder_pose_from_states(classify_encoder_states(thumb_five_positions, self.t)),
            'five_grasp',
        )
        self.assertEqual(
            classify_encoder_pose_from_states(classify_encoder_states(thumb_three_positions, self.t)),
            'three_grasp_b',
        )

    def test_click_requires_context_open_and_rising_edge(self):
        state = IndexClickDetectorState()

        base_positions = {1: 0.20, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10}
        press_positions = {1: 0.20, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.90}

        pose, state = detect_index_click_event(
            classify_encoder_states(base_positions, self.t), state, now_s=1.0, click_window_s=0.7
        )
        self.assertIsNone(pose)

        pose, state = detect_index_click_event(
            classify_encoder_states(press_positions, self.t), state, now_s=1.1, click_window_s=0.7
        )
        self.assertEqual(pose, 'index_single_click')

        pose, state = detect_index_click_event(
            classify_encoder_states(press_positions, self.t), state, now_s=1.2, click_window_s=0.7
        )
        self.assertIsNone(pose)

        no_context_positions = {1: 0.20, 2: 0.80, 3: 0.10, 4: 0.10, 5: 0.10}
        pose, state = detect_index_click_event(
            classify_encoder_states(no_context_positions, self.t), state, now_s=2.0, click_window_s=0.7
        )
        self.assertIsNone(pose)

    def test_click_double_click_window(self):
        state = IndexClickDetectorState()
        open_positions = {1: 0.20, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10}
        closed_positions = {1: 0.20, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.90}

        _, state = detect_index_click_event(
            classify_encoder_states(open_positions, self.t), state, now_s=0.0, click_window_s=0.7
        )
        pose, state = detect_index_click_event(
            classify_encoder_states(closed_positions, self.t), state, now_s=0.1, click_window_s=0.7
        )
        self.assertEqual(pose, 'index_single_click')

        _, state = detect_index_click_event(
            classify_encoder_states(open_positions, self.t), state, now_s=0.2, click_window_s=0.7
        )
        pose, state = detect_index_click_event(
            classify_encoder_states(closed_positions, self.t), state, now_s=0.4, click_window_s=0.7
        )
        self.assertEqual(pose, 'index_double_click')

    def test_mid_band_does_not_trigger_grasps(self):
        mid_positions = {1: 0.20, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50}
        states = classify_encoder_states(mid_positions, self.t)
        self.assertEqual(states['thumb_flex'], MID)
        self.assertEqual(states['ring'], MID)
        self.assertEqual(states['middle'], MID)
        self.assertEqual(states['index'], MID)
        self.assertIsNone(classify_encoder_pose_from_states(states))

    def test_state_labels_cover_open_mid_closed_and_thumb_modes(self):
        positions = {1: 0.65, 2: 0.10, 3: 0.50, 4: 0.90, 5: 0.10}
        states = classify_encoder_states(positions, self.t)
        self.assertEqual(states['thumb_swing'], IN)
        self.assertEqual(states['thumb_flex'], OPEN)
        self.assertEqual(states['ring'], MID)
        self.assertEqual(states['middle'], CLOSED)
        self.assertEqual(states['index'], OPEN)

        positions = {1: 0.20, 2: 0.10, 3: 0.10, 4: 0.10, 5: 0.10}
        states = classify_encoder_states(positions, self.t)
        self.assertEqual(states['thumb_swing'], NORMAL_OR_OUT)


if __name__ == '__main__':
    unittest.main()
