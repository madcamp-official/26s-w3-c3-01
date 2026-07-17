from __future__ import annotations

import unittest

import numpy as np

from cuecast_yolo.color_detector import ColorBallDetector
from cuecast_yolo.geometry import TableTransform
from cuecast_yolo.stop_detector import BallStopDetector


class TableTransformTest(unittest.TestCase):
    def test_rectangle_normalization(self) -> None:
        transform = TableTransform(
            corners=((100, 50), (1100, 50), (1100, 550), (100, 550))
        )
        x, y = transform.normalize((600, 300))
        self.assertAlmostEqual(x, 0.5, places=5)
        self.assertAlmostEqual(y, 0.5, places=5)

    def test_perspective_corners(self) -> None:
        transform = TableTransform(
            corners=((120, 50), (1080, 80), (1150, 560), (70, 530))
        )
        for source, expected in zip(
            transform.corners,
            ((0, 0), (1, 0), (1, 1), (0, 1)),
            strict=True,
        ):
            x, y = transform.normalize(source)
            self.assertAlmostEqual(x, expected[0], places=4)
            self.assertAlmostEqual(y, expected[1], places=4)


class BallStopDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.positions = {
            "white_ball": (0.1, 0.2),
            "yellow_ball": (0.7, 0.4),
            "red_ball": (0.4, 0.8),
        }

    def test_emits_once_after_stable_window(self) -> None:
        detector = BallStopDetector(stable_seconds=0.5)
        events = [
            detector.update(index * 0.1, self.positions) for index in range(11)
        ]
        self.assertEqual(sum(event is not None for event in events), 1)

    def test_emits_again_after_movement_and_new_stop(self) -> None:
        detector = BallStopDetector(stable_seconds=0.5)
        first = [detector.update(index * 0.1, self.positions) for index in range(6)]
        moved = {**self.positions, "red_ball": (0.5, 0.8)}
        second = [detector.update(0.6 + index * 0.1, moved) for index in range(7)]
        self.assertEqual(sum(event is not None for event in first + second), 2)

    def test_missing_ball_does_not_emit(self) -> None:
        detector = BallStopDetector(stable_seconds=0.5)
        incomplete = dict(self.positions)
        incomplete.pop("red_ball")
        events = [detector.update(index * 0.1, incomplete) for index in range(10)]
        self.assertTrue(all(event is None for event in events))


class ColorBallDetectorTest(unittest.TestCase):
    def test_rejects_table_touching_frame_edges(self) -> None:
        detector = ColorBallDetector()
        touching = np.float32([[0, 20], [639, 20], [639, 340], [0, 340]])
        self.assertFalse(detector._corners_have_frame_margin(touching, (360, 640, 3)))

    def test_accepts_rectangle_and_rejects_perspective_table(self) -> None:
        detector = ColorBallDetector()
        rectangle = np.int32([[[100, 80]], [[540, 80]], [[540, 300]], [[100, 300]]])
        trapezoid = np.int32([[[170, 60]], [[500, 90]], [[590, 310]], [[60, 330]]])
        self.assertTrue(detector._is_ceiling_geometry(rectangle))
        self.assertFalse(detector._is_ceiling_geometry(trapezoid))

    def test_surrounding_ratio_detects_blue_and_black_backgrounds(self) -> None:
        blue = np.full((80, 80), 255, dtype=np.uint8)
        dark = np.zeros((80, 80), dtype=np.uint8)
        blue_ratio, dark_ratio = ColorBallDetector._surrounding_ratios(
            40, 40, 14, 14, blue, dark
        )
        self.assertGreater(blue_ratio, 0.99)
        self.assertLess(dark_ratio, 0.01)

        blue[:, 40:] = 0
        dark[:, 40:] = 255
        blue_ratio, dark_ratio = ColorBallDetector._surrounding_ratios(
            40, 40, 14, 14, blue, dark
        )
        self.assertLess(blue_ratio, 0.62)
        self.assertGreater(dark_ratio, 0.20)


if __name__ == "__main__":
    unittest.main()
