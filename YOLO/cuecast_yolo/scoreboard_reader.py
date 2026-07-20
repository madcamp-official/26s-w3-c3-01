from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


DigitRecognizer = Callable[[np.ndarray], int | None]


@dataclass(frozen=True)
class ScoreboardReading:
    set_number: int
    inning: int
    player1_score: int
    player2_score: int
    player1_run: int
    player2_run: int

    def to_dict(self) -> dict[str, int]:
        return {
            "set": self.set_number,
            "inning": self.inning,
            "player1Score": self.player1_score,
            "player2Score": self.player2_score,
            "player1Run": self.player1_run,
            "player2Run": self.player2_run,
        }


class SyntheticDigitRecognizer:
    """Small dependency-free digit recognizer for the fixed PBA score graphic."""

    def __init__(self) -> None:
        self._samples, self._labels = self._make_training_set()
        self._knn = cv2.ml.KNearest_create()
        self._knn.train(self._samples, cv2.ml.ROW_SAMPLE, self._labels)

    @staticmethod
    def _feature(mask: np.ndarray) -> np.ndarray:
        points = cv2.findNonZero(mask)
        if points is None:
            return np.zeros((1, 20 * 32), np.float32)
        x, y, w, h = cv2.boundingRect(points)
        glyph = mask[y : y + h, x : x + w]
        scale = min(16 / max(w, 1), 28 / max(h, 1))
        resized = cv2.resize(
            glyph,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        canvas = np.zeros((32, 20), np.uint8)
        yy = (32 - resized.shape[0]) // 2
        xx = (20 - resized.shape[1]) // 2
        canvas[yy : yy + resized.shape[0], xx : xx + resized.shape[1]] = resized
        return (canvas.reshape(1, -1).astype(np.float32) / 255.0)

    @classmethod
    def _make_training_set(cls) -> tuple[np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        labels: list[float] = []
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        ]
        for digit in range(10):
            for font in fonts:
                for scale in (0.8, 0.9, 1.0, 1.1, 1.2):
                    for thickness in (2, 3, 4):
                        image = np.zeros((48, 36), np.uint8)
                        size, _ = cv2.getTextSize(str(digit), font, scale, thickness)
                        origin = ((36 - size[0]) // 2, (48 + size[1]) // 2)
                        cv2.putText(
                            image,
                            str(digit),
                            origin,
                            font,
                            scale,
                            255,
                            thickness,
                            cv2.LINE_AA,
                        )
                        features.append(cls._feature(image))
                        labels.append(float(digit))
        try:
            from PIL import Image, ImageDraw, ImageFont

            font_paths = [
                Path(path)
                for path in (
                    "C:/Windows/Fonts/arialbd.ttf",
                    "C:/Windows/Fonts/ARIALNB.TTF",
                    "C:/Windows/Fonts/seguisb.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                )
                if Path(path).exists()
            ]
            for digit in range(10):
                for font_path in font_paths:
                    for size in (28, 32, 36, 40):
                        for stroke in (0, 1):
                            canvas = Image.new("L", (48, 56), 0)
                            draw = ImageDraw.Draw(canvas)
                            font = ImageFont.truetype(str(font_path), size)
                            bounds = draw.textbbox(
                                (0, 0), str(digit), font=font, stroke_width=stroke
                            )
                            x = (48 - (bounds[2] - bounds[0])) // 2 - bounds[0]
                            y = (56 - (bounds[3] - bounds[1])) // 2 - bounds[1]
                            draw.text(
                                (x, y),
                                str(digit),
                                fill=255,
                                font=font,
                                stroke_width=stroke,
                                stroke_fill=255,
                            )
                            features.append(cls._feature(np.asarray(canvas)))
                            labels.append(float(digit))
        except ImportError:
            pass
        return np.vstack(features), np.asarray(labels, np.float32)

    def __call__(self, image: np.ndarray) -> int | None:
        if image.size == 0:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, normal = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        candidates = (normal, cv2.bitwise_not(normal))
        best: tuple[float, int] | None = None
        for mask in candidates:
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            glyphs: list[tuple[int, np.ndarray]] = []
            for x, y, w, h, area in stats[1:count]:
                if h < gray.shape[0] * 0.42 or area < 12:
                    continue
                if w > gray.shape[1] * 0.9 and h > gray.shape[0] * 0.9:
                    continue
                if w / max(h, 1) > 0.95:
                    continue
                touches_vertical_edge = x <= 1 or x + w >= gray.shape[1] - 1
                looks_like_cell_rule = w <= max(3, round(h * 0.22))
                if touches_vertical_edge and looks_like_cell_rule:
                    continue
                glyphs.append((x, mask[y : y + h, x : x + w]))
            if not 1 <= len(glyphs) <= 3:
                continue
            digits: list[str] = []
            distances_for_number: list[float] = []
            for _, glyph in sorted(glyphs, key=lambda item: item[0]):
                feature = self._feature(glyph)
                _, result, _, distances = self._knn.findNearest(feature, k=3)
                distances_for_number.append(float(distances.mean()))
                digits.append(str(int(result[0, 0])))
            distance = sum(distances_for_number) / len(distances_for_number)
            value = int("".join(digits))
            if best is None or distance < best[0]:
                best = (distance, value)
        return None if best is None else best[1]


class PbaScoreboardReader:
    """Read the stable lower-left PBA/LPBA score graphic by semantic cell."""

    # Coordinates are relative to the black score panel, excluding the sponsor strip.
    DIGIT_CELLS = {
        "set_number": (0.715, 0.02, 0.795, 0.27),
        "inning": (0.865, 0.02, 0.955, 0.27),
        "player1_score": (0.715, 0.29, 0.825, 0.64),
        "player2_score": (0.715, 0.65, 0.825, 0.99),
        "player1_run": (0.855, 0.36, 0.910, 0.60),
        "player2_run": (0.855, 0.72, 0.910, 0.96),
    }

    def __init__(self, recognizer: DigitRecognizer | None = None) -> None:
        self.recognizer = recognizer or SyntheticDigitRecognizer()

    @staticmethod
    def _lower_left(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        height, width = frame.shape[:2]
        x2, y1 = width, round(height * 0.30)
        return frame[y1:height, 0:x2], 0, y1

    @classmethod
    def _locate_from_score_cells(
        cls, frame: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        search, offset_x, offset_y = cls._lower_left(frame)
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        bright = cv2.inRange(gray, 175, 255)
        bright = cv2.morphologyEx(
            bright,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_h, frame_w = frame.shape[:2]
        # The two score backgrounds touch and normally form one tall bright block.
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour) / max(w * h, 1)
            if not (0.025 * frame_w <= w <= 0.12 * frame_w):
                continue
            if not (0.20 * frame_h <= h <= 0.50 * frame_h):
                continue
            if not 0.35 <= w / max(h, 1) <= 0.75 or fill < 0.75:
                continue
            panel_w = round(w / 0.096)
            panel_h = round(h / 0.648)
            panel_x = round(x + offset_x - 0.72 * panel_w)
            panel_y = round(y + offset_y - 0.316 * panel_h)
            if 0 <= panel_x <= 0.15 * frame_w and panel_y >= 0:
                if panel_x + panel_w <= frame_w and panel_y + panel_h <= frame_h:
                    return panel_x, panel_y, panel_w, panel_h
        cells: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (0.025 * frame_w <= w <= 0.12 * frame_w):
                continue
            if not (0.05 * frame_h <= h <= 0.22 * frame_h):
                continue
            if not 0.65 <= w / max(h, 1) <= 1.45:
                continue
            if cv2.contourArea(contour) / max(w * h, 1) < 0.65:
                continue
            cells.append((x + offset_x, y + offset_y, w, h))
        for first in cells:
            for second in cells:
                if second[1] <= first[1]:
                    continue
                if abs(first[0] - second[0]) > max(first[2], second[2]) * 0.25:
                    continue
                if abs(first[2] - second[2]) > max(first[2], second[2]) * 0.25:
                    continue
                if not 0.7 * first[3] <= second[3] <= 1.3 * first[3]:
                    continue
                panel_w = round(((first[2] + second[2]) / 2) / 0.11)
                panel_h = round(((first[3] + second[3]) / 2) / 0.35)
                panel_x = round(first[0] - 0.715 * panel_w)
                panel_y = round(first[1] - 0.29 * panel_h)
                if panel_x < 0 or panel_y < 0:
                    continue
                if panel_x + panel_w > frame_w or panel_y + panel_h > frame_h:
                    continue
                return panel_x, panel_y, panel_w, panel_h
        return None

    @classmethod
    def locate_panel(cls, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        search, offset_x, offset_y = cls._lower_left(frame)
        gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(gray, 0, 38)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_h, frame_w = frame.shape[:2]
        matches: list[tuple[float, tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if not (0.16 * frame_w <= w <= 0.46 * frame_w):
                continue
            if not (0.07 * frame_h <= h <= 0.28 * frame_h):
                continue
            if not 2.5 <= aspect <= 5.3:
                continue
            fill = cv2.contourArea(contour) / max(w * h, 1)
            if fill < 0.45:
                continue
            matches.append((w * h * fill, (x + offset_x, y + offset_y, w, h)))
        located = max(matches, default=(0.0, None), key=lambda item: item[0])[1]
        return located or cls._locate_from_score_cells(frame)

    @staticmethod
    def _crop(panel: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        h, w = panel.shape[:2]
        x1, y1, x2, y2 = box
        return panel[round(y1 * h) : round(y2 * h), round(x1 * w) : round(x2 * w)]

    def read(self, frame: np.ndarray) -> ScoreboardReading | None:
        located = self.locate_panel(frame)
        if located is None:
            return None
        x, y, w, h = located
        panel = frame[y : y + h, x : x + w]
        values = {
            name: self.recognizer(self._crop(panel, box))
            for name, box in self.DIGIT_CELLS.items()
        }
        required = ("set_number", "inning", "player1_score", "player2_score")
        if any(values[name] is None for name in required):
            return None
        return ScoreboardReading(
            set_number=int(values["set_number"]),
            inning=int(values["inning"]),
            player1_score=int(values["player1_score"]),
            player2_score=int(values["player2_score"]),
            player1_run=int(values["player1_run"] or 0),
            player2_run=int(values["player2_run"] or 0),
        )


class StableScoreboardState:
    """Confirm OCR only after repeated frames and reject impossible regressions."""

    def __init__(self, confirmations: int = 3, history_size: int = 5) -> None:
        self.confirmations = confirmations
        self._history: deque[ScoreboardReading] = deque(maxlen=history_size)
        self.current: ScoreboardReading | None = None

    def update(self, reading: ScoreboardReading | None) -> ScoreboardReading | None:
        if reading is None:
            return None
        self._history.append(reading)
        candidate, count = Counter(self._history).most_common(1)[0]
        if count < self.confirmations or candidate == self.current:
            return None
        if self.current is not None:
            same_set = candidate.set_number == self.current.set_number
            if candidate.set_number < self.current.set_number:
                return None
            if same_set and (
                candidate.inning < self.current.inning
                or candidate.player1_score < self.current.player1_score
                or candidate.player2_score < self.current.player2_score
            ):
                return None
        self.current = candidate
        self._history.clear()
        return candidate
