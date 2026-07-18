from __future__ import annotations

import unittest

from cuecast_yolo.video_position_analyzer import VideoPositionAnalyzer


class VideoPositionAnalyzerTest(unittest.TestCase):
    def test_rejects_negative_video_position_before_opening_source(self) -> None:
        analyzer = VideoPositionAnalyzer()
        with self.assertRaisesRegex(ValueError, "0초 이상"):
            analyzer.analyze("unused", -0.1)

    def test_rejects_excessive_lookback_before_opening_source(self) -> None:
        analyzer = VideoPositionAnalyzer()
        with self.assertRaisesRegex(ValueError, "2~30초"):
            analyzer.analyze("unused", 10.0, lookback_seconds=31.0)


if __name__ == "__main__":
    unittest.main()
