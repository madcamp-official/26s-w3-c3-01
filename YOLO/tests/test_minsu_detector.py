from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from cuecast_yolo.detector import BallDetection
from cuecast_yolo.minsu_detector import MinsuRealtimeDetector, valid_inner_corners


OUTER_CLOTH = np.float32(
    [[100, 50], [540, 50], [540, 310], [100, 310]]
)
INNER_CUSHION = np.float32(
    [[120, 70], [520, 70], [520, 290], [120, 290]]
)


class _FakeBallDetector:
    def detect(self, _frame: np.ndarray) -> dict[str, BallDetection]:
        centers = {
            "white_ball": (120.0, 70.0),
            "yellow_ball": (520.0, 290.0),
            "red_ball": (320.0, 180.0),
        }
        return {
            name: BallDetection(name, 0.9, (0, 0, 1, 1), center)
            for name, center in centers.items()
        }


def _detector_without_model() -> MinsuRealtimeDetector:
    detector = MinsuRealtimeDetector.__new__(MinsuRealtimeDetector)
    detector.detector = _FakeBallDetector()
    detector.view_corners = None
    detector.reference_corners = None
    return detector


class MinsuInnerTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def test_inner_boundary_is_used_for_ball_coordinates(self) -> None:
        detector = _detector_without_model()
        with (
            patch(
                "cuecast_yolo.minsu_detector.detect_corners_fast",
                return_value=OUTER_CLOTH,
            ),
            patch(
                "cuecast_yolo.minsu_detector.find_table_corners",
                return_value=OUTER_CLOTH,
            ),
            patch(
                "cuecast_yolo.minsu_detector.detect_inner_table_corners",
                return_value=INNER_CUSHION,
            ),
        ):
            result = detector.detect(self.frame)

        self.assertTrue(result.valid_view)
        np.testing.assert_allclose(detector.view_corners, OUTER_CLOTH)
        np.testing.assert_allclose(detector.reference_corners, INNER_CUSHION)
        np.testing.assert_allclose(result.positions["white_ball"], (0.0, 0.0))
        np.testing.assert_allclose(result.positions["yellow_ball"], (1.0, 1.0))
        np.testing.assert_allclose(result.positions["red_ball"], (0.5, 0.5))

    def test_outer_cloth_is_not_used_when_inner_boundary_is_invalid(self) -> None:
        detector = _detector_without_model()
        invalid_inner = np.float32(
            [[40, 20], [600, 20], [600, 340], [40, 340]]
        )
        with (
            patch(
                "cuecast_yolo.minsu_detector.detect_corners_fast",
                return_value=OUTER_CLOTH,
            ),
            patch(
                "cuecast_yolo.minsu_detector.find_table_corners",
                return_value=OUTER_CLOTH,
            ),
            patch(
                "cuecast_yolo.minsu_detector.detect_inner_table_corners",
                return_value=invalid_inner,
            ),
        ):
            result = detector.detect(self.frame)

        self.assertFalse(result.valid_view)
        self.assertIsNone(detector.reference_corners)
        self.assertFalse(result.positions)

    def test_inner_boundary_must_stay_inside_segmented_view(self) -> None:
        self.assertTrue(valid_inner_corners(INNER_CUSHION, OUTER_CLOTH, self.frame.shape))
        shifted = INNER_CUSHION + np.float32([[-80, 0]])
        self.assertFalse(valid_inner_corners(shifted, OUTER_CLOTH, self.frame.shape))


if __name__ == "__main__":
    unittest.main()
