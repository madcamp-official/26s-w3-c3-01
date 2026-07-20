from __future__ import annotations

import unittest

from cuecast_yolo.realtime_tracker import RealtimeLayoutTracker


LAYOUT = {
    "white_ball": (0.2, 0.3),
    "yellow_ball": (0.5, 0.4),
    "red_ball": (0.8, 0.7),
}
CONFIDENCE = {name: 0.9 for name in LAYOUT}


class RealtimeLayoutTrackerTest(unittest.TestCase):
    def test_preview_is_available_before_confirmation(self) -> None:
        tracker = RealtimeLayoutTracker(stable_seconds=0.6)
        update = tracker.update(0.0, LAYOUT, CONFIDENCE, valid_view=True)
        self.assertEqual(update.state, "settling")
        self.assertEqual(set(update.positions), set(LAYOUT))
        self.assertIsNone(update.confirmed_positions)

    def test_stable_layout_is_confirmed_once(self) -> None:
        tracker = RealtimeLayoutTracker(stable_seconds=0.6)
        tracker.update(0.0, LAYOUT, CONFIDENCE, valid_view=True)
        tracker.update(0.3, LAYOUT, CONFIDENCE, valid_view=True)
        confirmed = tracker.update(0.6, LAYOUT, CONFIDENCE, valid_view=True)
        self.assertEqual(confirmed.state, "confirmed")
        self.assertIsNotNone(confirmed.confirmed_positions)
        repeated = tracker.update(0.8, LAYOUT, CONFIDENCE, valid_view=True)
        self.assertIsNone(repeated.confirmed_positions)

    def test_camera_cut_freezes_last_positions(self) -> None:
        tracker = RealtimeLayoutTracker()
        tracker.update(0.0, LAYOUT, CONFIDENCE, valid_view=True)
        cut = tracker.update(0.2, {}, {}, valid_view=False)
        self.assertEqual(cut.state, "camera_cut")
        self.assertEqual(set(cut.positions), set(LAYOUT))
        self.assertEqual(cut.cut_positions, LAYOUT)
        repeated = tracker.update(0.4, {}, {}, valid_view=False)
        self.assertIsNone(repeated.cut_positions)

    def test_missing_detection_is_held_briefly(self) -> None:
        tracker = RealtimeLayoutTracker(missing_hold_seconds=0.45)
        tracker.update(0.0, LAYOUT, CONFIDENCE, valid_view=True)
        held = tracker.update(0.3, {}, {}, valid_view=True)
        expired = tracker.update(0.6, {}, {}, valid_view=True)
        self.assertEqual(set(held.positions), set(LAYOUT))
        self.assertFalse(expired.positions)


if __name__ == "__main__":
    unittest.main()
