from __future__ import annotations

import unittest

import cv2
import numpy as np

from cuecast_yolo.scoreboard_reader import (
    PbaScoreboardReader,
    ScoreboardReading,
    StableScoreboardState,
    SyntheticDigitRecognizer,
)


class PbaScoreboardReaderTest(unittest.TestCase):
    def test_digit_recognizer_reads_two_digit_values(self) -> None:
        image = np.full((52, 62, 3), 245, np.uint8)
        cv2.putText(
            image,
            "15",
            (5, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        self.assertEqual(SyntheticDigitRecognizer()(image), 15)

    def test_digit_recognizer_ignores_thin_score_cell_border(self) -> None:
        image = np.full((15, 18, 3), 245, np.uint8)
        cv2.rectangle(image, (0, 0), (2, 14), (10, 10, 10), -1)
        cv2.putText(
            image,
            "0",
            (6, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        self.assertEqual(SyntheticDigitRecognizer()(image), 0)

    def test_digit_recognizer_keeps_interior_one(self) -> None:
        image = np.full((20, 18, 3), 245, np.uint8)
        cv2.putText(
            image,
            "1",
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )
        self.assertEqual(SyntheticDigitRecognizer()(image), 1)

    def test_maps_each_fixed_cell_to_its_semantic_value(self) -> None:
        frame = np.full((720, 1280, 3), 140, np.uint8)
        cv2.rectangle(frame, (25, 535), (500, 675), (8, 8, 8), -1)
        expected = iter((2, 6, 5, 5, 2, None))
        reader = PbaScoreboardReader(lambda _image: next(expected))

        result = reader.read(frame)

        self.assertEqual(
            result,
            ScoreboardReading(2, 6, 5, 5, 2, 0),
        )

    def test_requires_three_matching_frames(self) -> None:
        state = StableScoreboardState(confirmations=3)
        reading = ScoreboardReading(2, 6, 5, 5, 2, 0)
        self.assertIsNone(state.update(reading))
        self.assertIsNone(state.update(reading))
        self.assertEqual(state.update(reading), reading)

    def test_rejects_score_regression_in_same_set(self) -> None:
        state = StableScoreboardState(confirmations=1)
        state.update(ScoreboardReading(2, 6, 5, 5, 2, 0))
        result = state.update(ScoreboardReading(2, 7, 4, 5, 0, 0))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
