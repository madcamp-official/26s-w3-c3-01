from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from cuecast_yolo.live_youtube import YoutubeLiveWorker, layout_distance
from cuecast_yolo.minsu_detector import TrackingFrame
from cuecast_yolo.scoreboard_reader import ScoreboardReading
from cuecast_yolo.video_position_analyzer import VideoPositionAnalyzer


POSITIONS = {
    "white_ball": (0.1, 0.2),
    "yellow_ball": (0.4, 0.5),
    "red_ball": (0.7, 0.8),
}


class YoutubeLiveWorkerTest(unittest.TestCase):
    def test_realtime_analyzer_prefers_first_top_view_fixed_yolo_detector(self) -> None:
        analyzer = VideoPositionAnalyzer.__new__(VideoPositionAnalyzer)
        analyzer._detector_lock = MagicMock()
        analyzer.minsu_detector = MagicMock()
        expected = TrackingFrame(POSITIONS, {name: 0.9 for name in POSITIONS}, True)
        analyzer.minsu_detector.detect.return_value = expected
        frame = np.zeros((360, 640, 3), dtype=np.uint8)

        actual = analyzer.detect_tracking_frame(frame)

        self.assertIs(actual, expected)
        analyzer.minsu_detector.detect.assert_called_once_with(frame)

    def test_layout_distance_uses_largest_ball_movement(self) -> None:
        moved = {**POSITIONS, "red_ball": (0.73, 0.84)}
        self.assertAlmostEqual(layout_distance(POSITIONS, moved), 0.05)

    def test_rejects_invalid_shooter(self) -> None:
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "white 또는 yellow"):
            worker.set_shooter("red")

    def test_sync_request_is_consumed_once(self) -> None:
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), lambda *_: None)
        worker._status["running"] = True

        worker.sync_to(12.5)

        self.assertEqual(worker.status()["requestedSyncSeconds"], 12.5)
        self.assertTrue(worker.status()["syncing"])
        self.assertEqual(worker._take_pending_sync(), 12.5)
        self.assertIsNone(worker._take_pending_sync())

    def test_sync_rejects_negative_time(self) -> None:
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "0 이상"):
            worker.sync_to(-0.1)

    def test_scoreboard_active_circle_automatically_sets_shooter(self) -> None:
        layout_callback = MagicMock()
        callback = MagicMock()
        worker = YoutubeLiveWorker(
            VideoPositionAnalyzer(), layout_callback, scoreboard_callback=callback
        )
        reading = ScoreboardReading(1, 2, 3, 4, 0, 1, "yellow", "white")

        worker._accept_scoreboard(reading, 12.5)

        status = worker.status()
        self.assertEqual(status["shooter"], "yellow")
        self.assertTrue(status["shooterConfirmed"])
        self.assertEqual(status["scoreboard"]["activeColor"], "yellow")
        callback.assert_called_once()
        layout_callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
