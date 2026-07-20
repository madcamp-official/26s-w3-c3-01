from __future__ import annotations

import unittest

from cuecast_yolo.live_youtube import YoutubeLiveWorker, layout_distance
from cuecast_yolo.video_position_analyzer import VideoPositionAnalyzer


POSITIONS = {
    "white_ball": (0.1, 0.2),
    "yellow_ball": (0.4, 0.5),
    "red_ball": (0.7, 0.8),
}


class YoutubeLiveWorkerTest(unittest.TestCase):
    def test_layout_distance_uses_largest_ball_movement(self) -> None:
        moved = {**POSITIONS, "red_ball": (0.73, 0.84)}
        self.assertAlmostEqual(layout_distance(POSITIONS, moved), 0.05)

    def test_rejects_invalid_shooter(self) -> None:
        worker = YoutubeLiveWorker(VideoPositionAnalyzer(), lambda *_: None)
        with self.assertRaisesRegex(ValueError, "white 또는 yellow"):
            worker.set_shooter("red")


if __name__ == "__main__":
    unittest.main()
