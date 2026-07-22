from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cuecast_yolo.live_youtube import YoutubeLiveWorker, layout_distance
from cuecast_yolo.minsu_detector import TrackingFrame
from cuecast_yolo.scoreboard_reader import ScoreboardReading
from cuecast_yolo.video_position_analyzer import VideoPositionAnalyzer, VideoSource


POSITIONS = {
    "white_ball": (0.1, 0.2),
    "yellow_ball": (0.4, 0.5),
    "red_ball": (0.7, 0.8),
}


class YoutubeLiveWorkerTest(unittest.TestCase):
    @patch("cuecast_yolo.live_youtube.cv2.VideoCapture")
    def test_refreshes_youtube_cdn_url_when_cached_stream_cannot_open(
        self, video_capture: MagicMock
    ) -> None:
        analyzer = MagicMock()
        analyzer.resolve.side_effect = [
            VideoSource("stale-url", "match", 100.0, "youtube"),
            VideoSource("fresh-url", "match", 100.0, "youtube"),
        ]
        stale_capture = MagicMock()
        stale_capture.isOpened.return_value = False
        fresh_capture = MagicMock()
        fresh_capture.isOpened.return_value = True
        video_capture.side_effect = [stale_capture, fresh_capture]
        worker = YoutubeLiveWorker(analyzer, lambda *_: None)

        video, capture = worker._open_capture("youtube-url")

        self.assertEqual(video.media_url, "fresh-url")
        self.assertIs(capture, fresh_capture)
        stale_capture.release.assert_called_once_with()
        analyzer.invalidate_resolved_source.assert_called_once_with("youtube-url")

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

    def test_manual_shooter_updates_status_and_republishes_layout(self) -> None:
        layout_callback = MagicMock()
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), layout_callback)
        worker._publish(
            POSITIONS,
            timestamp=10.0,
            source="stopped",
            confidence=0.9,
            state="confirmed",
            confirmed=True,
        )
        layout_callback.reset_mock()

        worker.set_shooter("yellow")

        status = worker.status()
        self.assertEqual(status["shooter"], "yellow")
        self.assertTrue(status["shooterConfirmed"])
        self.assertEqual(status["shooterSource"], "manual")
        layout_callback.assert_called_once()
        self.assertEqual(layout_callback.call_args.args[1], "yellow")
        self.assertTrue(layout_callback.call_args.args[2]["shooterRefresh"])

    def test_reset_scoreboard_clears_ocr_status(self) -> None:
        reader = MagicMock()
        reader.enabled = True
        worker = YoutubeLiveWorker(
            VideoPositionAnalyzer(), lambda *_: None, scoreboard_reader=reader
        )
        worker._status.update(
            scoreboardDetected=True,
            scoreboard={"set": 3},
            shooterConfirmed=True,
            shooterSource="scoreboard",
        )
        worker._shooter_confirmed = True

        worker.reset_scoreboard()

        reader.reset.assert_called_once_with()
        status = worker.status()
        self.assertFalse(status["scoreboardDetected"])
        self.assertIsNone(status["scoreboard"])
        self.assertFalse(status["shooterConfirmed"])

    def test_fast_scoreboard_color_confirms_shooter_before_full_ocr(self) -> None:
        layout_callback = MagicMock()
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), layout_callback)
        worker._publish(
            POSITIONS,
            timestamp=10.0,
            source="stopped",
            confidence=0.9,
            state="confirmed",
            confirmed=True,
        )
        layout_callback.reset_mock()

        worker._accept_fast_shooter("yellow", 10.25)

        status = worker.status()
        self.assertEqual(status["shooter"], "yellow")
        self.assertTrue(status["shooterConfirmed"])
        self.assertEqual(status["shooterSource"], "scoreboard_fast")
        layout_callback.assert_called_once()

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
        reading = ScoreboardReading(1, 2, 3, 4, None, 1, "yellow", "white")

        worker._accept_scoreboard(reading, 12.5)

        status = worker.status()
        self.assertEqual(status["shooter"], "yellow")
        self.assertTrue(status["shooterConfirmed"])
        self.assertEqual(status["scoreboard"]["activeColor"], "yellow")
        self.assertIsNone(status["scoreboard"]["player1Run"])
        self.assertEqual(status["scoreboard"]["player2Run"], 1)
        callback.assert_called_once()
        layout_callback.assert_not_called()

    def test_only_active_player_keeps_a_run_value(self) -> None:
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), lambda *_: None)
        worker._accept_scoreboard(
            ScoreboardReading(1, 2, 3, 4, 2, None, "white", "white"), 12.5
        )
        first = worker.status()["scoreboard"]
        self.assertEqual(first["player1Run"], 2)
        self.assertIsNone(first["player2Run"])

        worker._accept_scoreboard(
            ScoreboardReading(None, None, None, None, None, 3, "yellow", "white"),
            13.0,
        )
        second = worker.status()["scoreboard"]
        self.assertIsNone(second["player1Run"])
        self.assertEqual(second["player2Run"], 3)

    def test_ambiguous_two_run_values_are_hidden(self) -> None:
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), lambda *_: None)
        worker._accept_scoreboard(
            ScoreboardReading(1, 2, 3, 4, 1, 2, "white", "white"), 12.5
        )
        scoreboard = worker.status()["scoreboard"]
        self.assertIsNone(scoreboard["player1Run"])
        self.assertIsNone(scoreboard["player2Run"])

    def test_partial_scoreboard_updates_are_merged_before_callback(self) -> None:
        callback = MagicMock()
        worker = YoutubeLiveWorker(
            VideoPositionAnalyzer(), lambda *_: None, scoreboard_callback=callback
        )

        worker._accept_scoreboard(
            ScoreboardReading(None, None, 3, 4, None, None, None, "white"),
            12.0,
        )
        worker._accept_scoreboard(
            ScoreboardReading(1, None, None, None, None, None, None, "white", "A", "B"),
            12.5,
        )

        scoreboard = worker.status()["scoreboard"]
        self.assertEqual(scoreboard["set"], 1)
        self.assertEqual(scoreboard["player1Score"], 3)
        self.assertEqual(scoreboard["player2Score"], 4)
        self.assertEqual(scoreboard["player1Name"], "A")
        self.assertEqual(callback.call_args.args[0], scoreboard)

    def test_scoreboard_confirm_republishes_last_confirmed_layout(self) -> None:
        layout_callback = MagicMock()
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), layout_callback)
        # 공이 멈춰 레이아웃이 먼저 확정되었지만 아직 수구는 미확정 상태.
        worker._publish(
            POSITIONS,
            timestamp=10.0,
            source="stopped",
            confidence=0.9,
            confidences={name: 0.9 for name in POSITIONS},
            state="confirmed",
            confirmed=True,
        )
        first_analysis = layout_callback.call_args.args[2]
        self.assertFalse(first_analysis["shooterConfirmed"])
        layout_callback.reset_mock()

        # 점수판이 원형 안 숫자 색으로 수구를 확정하는 순간 즉시 재예측되어야 한다.
        reading = ScoreboardReading(1, 2, 3, 4, 0, 1, "yellow", "white")
        worker._accept_scoreboard(reading, 12.5)

        layout_callback.assert_called_once()
        positions, shooter, analysis = layout_callback.call_args.args
        self.assertEqual(shooter, "yellow")
        self.assertTrue(analysis["shooterConfirmed"])
        self.assertTrue(analysis["confirmed"])
        self.assertTrue(analysis["shooterRefresh"])

    def test_scoreboard_confirm_without_prior_layout_does_not_republish(self) -> None:
        layout_callback = MagicMock()
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), layout_callback)
        worker._accept_scoreboard(
            ScoreboardReading(1, 2, 3, 4, 0, 1, "yellow", "white"), 12.5
        )
        layout_callback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
