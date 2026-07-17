from __future__ import annotations

import unittest

import cv2
import numpy as np

from create_virtual_comparison import fit_to_panel, to_virtual_coordinates
from cuecast_yolo.color_detector import ColorBallDetector
from cuecast_yolo.geometry import TableTransform
from cuecast_yolo.precut import PreCutLayoutBuffer
from cuecast_yolo.stop_detector import BallStopDetector
from cuecast_yolo.view_gate import fixed_top_view_ratios, is_fixed_top_view


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


class FixedTopViewGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.corners = np.int32([[30, 20], [170, 20], [170, 80], [30, 80]])

    def test_accepts_blue_inside_and_non_blue_outside(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.fillConvexPoly(frame, self.corners, (255, 150, 40))
        inner, outer = fixed_top_view_ratios(frame, self.corners, ring_width=8)
        self.assertGreater(inner, 0.95)
        self.assertLess(outer, 0.05)
        self.assertTrue(is_fixed_top_view(frame, self.corners, ring_width=8))

    def test_rejects_zoomed_blue_table_outside_fixed_corners(self) -> None:
        frame = np.full((100, 200, 3), (255, 150, 40), dtype=np.uint8)
        inner, outer = fixed_top_view_ratios(frame, self.corners, ring_width=8)
        self.assertGreater(inner, 0.95)
        self.assertGreater(outer, 0.95)
        self.assertFalse(is_fixed_top_view(frame, self.corners, ring_width=8))


class VirtualComparisonTest(unittest.TestCase):
    def test_converts_detector_names_to_virtual_renderer_format(self) -> None:
        converted = to_virtual_coordinates(
            {
                "white_ball": (0.1, 0.2),
                "yellow_ball": (0.3, 0.4),
                "red_ball": (0.5, 0.6),
            }
        )
        self.assertEqual(converted["white"]["normalized"]["x"], 0.1)
        self.assertEqual(converted["yellow"]["normalized"]["y"], 0.4)
        self.assertEqual(converted["red"]["normalized"]["x"], 0.5)

    def test_fits_virtual_table_into_video_panel(self) -> None:
        source = np.full((800, 1400, 3), 255, dtype=np.uint8)
        panel = fit_to_panel(source, 640, 360)
        self.assertEqual(panel.shape, (360, 640, 3))


class PreCutLayoutBufferTest(unittest.TestCase):
    @staticmethod
    def positions(offset: float) -> dict[str, tuple[float, float]]:
        return {
            "white_ball": (0.2 + offset, 0.2),
            "yellow_ball": (0.5 + offset, 0.5),
            "red_ball": (0.8 + offset, 0.8),
        }

    def test_finalizes_median_of_nearly_stopped_samples(self) -> None:
        buffer = PreCutLayoutBuffer(max_step=0.02, max_span=0.04)
        buffer.add(0.0, self.positions(0.000))
        buffer.add(0.1, self.positions(0.010))
        buffer.add(0.2, self.positions(0.015))
        event = buffer.finalize_on_cut(0.3)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertAlmostEqual(event.positions["white_ball"][0], 0.210)

    def test_rejects_fast_movement_before_cut(self) -> None:
        buffer = PreCutLayoutBuffer(max_step=0.02, max_span=0.04)
        buffer.add(0.0, self.positions(0.00))
        buffer.add(0.1, self.positions(0.03))
        buffer.add(0.2, self.positions(0.06))
        self.assertIsNone(buffer.finalize_on_cut(0.3))


if __name__ == "__main__":
    unittest.main()
