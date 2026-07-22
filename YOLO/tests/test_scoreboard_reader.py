from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cuecast_yolo.scoreboard_reader import (
    FastPbaCueColorReader,
    PbaScoreboardReader,
    RealtimePbaScoreboardReader,
    ScoreboardReading,
    StableScoreboardState,
    SyntheticDigitRecognizer,
    TesseractNameRecognizer,
)


class _SequenceRecognizer:
    available = True

    def __init__(self, values: list[int]) -> None:
        self.values = iter(values)

    def __call__(self, _image, _modes=(8, 7, 13)) -> int:
        return next(self.values)


class _SequenceNameRecognizer:
    def __init__(self, values: list[str]) -> None:
        self.values = iter(values)
        self.calls = 0

    def __call__(self, _image) -> str:
        self.calls += 1
        return next(self.values)


class PbaScoreboardReaderTest(unittest.TestCase):
    def test_fast_cue_reader_confirms_color_without_digit_ocr(self) -> None:
        reader = FastPbaCueColorReader()
        reader.box_white = (10, 10, 10, 10)
        reader.box_yellow = (10, 30, 10, 10)
        reader.circle_white = (30, 10, 10, 10)
        reader.circle_yellow = (30, 30, 10, 10)
        frame = np.zeros((60, 60, 3), np.uint8)

        with patch.object(
            reader,
            "_has_circle_digit",
            side_effect=lambda _frame, _box, color: color == "yellow",
        ):
            first = reader.sample(1, frame)
            confirmed = reader.sample(2, frame)
            unchanged = reader.sample(3, frame)

        self.assertIsNone(first)
        self.assertEqual(confirmed, "yellow")
        self.assertIsNone(unchanged)

    def test_name_normalizer_keeps_korean_and_english_letters(self) -> None:
        self.assertEqual(
            TesseractNameRecognizer.normalize("  김재근  12 / KIM  "),
            "김재근 KIM",
        )

    def test_reading_serializes_ocr_player_names(self) -> None:
        reading = ScoreboardReading(
            1, 3, 5, 2, 2, 0, "white", "white", "김영원", "김재근"
        )
        self.assertEqual(reading.to_dict()["player1Name"], "김영원")
        self.assertEqual(reading.to_dict()["player2Name"], "김재근")

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

    def test_realtime_reader_confirms_scores_and_active_cue_twice(self) -> None:
        reader = RealtimePbaScoreboardReader(
            _SequenceRecognizer([]),
            _SequenceRecognizer([1, 3, 1, 3, 1, 3, 1, 3]),
        )
        reader.box_white = (70, 30, 11, 14)
        reader.box_yellow = (70, 44, 11, 14)
        reader.circle_white = (85, 30, 12, 12)
        reader.circle_yellow = (85, 44, 12, 12)
        reader.panel_box = (0, 0, 100, 60)
        frame = np.full((80, 120, 3), 128, np.uint8)

        def score(_frame, _box, color):
            return 5 if color == "white" else 2

        def circle(_frame, _box, color):
            return (2, True) if color == "white" else (None, False)

        with (
            patch.object(reader, "_read_colored", side_effect=score),
            patch.object(reader, "_read_circle", side_effect=circle),
        ):
            self.assertIsNone(reader.sample(1, frame))
            self.assertIsNone(reader.sample(2, frame))
            self.assertIsNone(reader.sample(3, frame))
            result = reader.sample(4, frame)

        self.assertEqual(
            result,
            ScoreboardReading(1, 3, 5, 2, 2, 0, "white", "white"),
        )
        self.assertEqual(reader._committed["inning"], 3)

    def test_realtime_reader_requires_three_matching_name_reads(self) -> None:
        name_recognizer = _SequenceNameRecognizer(
            ["김영원", "김재근", "김영원", "김재근", "김영원", "김재근"]
        )
        reader = RealtimePbaScoreboardReader(
            _SequenceRecognizer([]),
            _SequenceRecognizer([1, 3, 1, 3, 1, 3, 1, 3, 1, 3]),
            name_recognizer,
        )
        reader.box_white = (70, 30, 11, 14)
        reader.box_yellow = (70, 44, 11, 14)
        reader.circle_white = (85, 30, 12, 12)
        reader.circle_yellow = (85, 44, 12, 12)
        reader.panel_box = (0, 0, 100, 60)
        frame = np.full((80, 120, 3), 128, np.uint8)

        def score(_frame, _box, color):
            return 5 if color == "white" else 2

        def circle(_frame, _box, color):
            return (2, True) if color == "white" else (None, False)

        with (
            patch.object(reader, "_read_colored", side_effect=score),
            patch.object(reader, "_read_circle", side_effect=circle),
        ):
            first = reader.sample(1, frame)
            second = reader.sample(2, frame)
            third = reader.sample(3, frame)
            result = reader.sample(4, frame)
            after_lock = reader.sample(5, frame)

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNone(third)
        self.assertEqual(result.player1_name, "김영원")
        self.assertEqual(result.player2_name, "김재근")
        self.assertTrue(reader.names_locked)
        self.assertIsNone(after_lock)
        self.assertEqual(name_recognizer.calls, 6)

    def test_realtime_reader_rejects_impossible_set_eight_then_accepts_three(self) -> None:
        header_values = [value for pair in ([(8, 1)] * 4 + [(3, 1)] * 4) for value in pair]
        reader = RealtimePbaScoreboardReader(
            _SequenceRecognizer([]),
            _SequenceRecognizer(header_values),
            lambda _image: None,
        )
        reader.box_white = (70, 30, 11, 14)
        reader.box_yellow = (70, 44, 11, 14)
        reader.circle_white = (85, 30, 12, 12)
        reader.circle_yellow = (85, 44, 12, 12)
        reader.panel_box = (0, 0, 100, 60)
        frame = np.full((80, 120, 3), 128, np.uint8)

        with (
            patch.object(
                reader,
                "_read_colored",
                side_effect=lambda _frame, _box, color: 5 if color == "white" else 2,
            ),
            patch.object(
                reader,
                "_read_circle",
                side_effect=lambda _frame, _box, color: (
                    (2, True) if color == "white" else (None, False)
                ),
            ),
        ):
            rejected = []
            accepted = []
            for index in range(1, 9):
                reader._signature = None
                result = reader.sample(index, frame)
                (rejected if index <= 4 else accepted).append(result)

        self.assertTrue(all(result is None for result in rejected))
        self.assertEqual(accepted[-1].set_number, 3)


if __name__ == "__main__":
    unittest.main()
