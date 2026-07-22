from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable
import unicodedata

import cv2
import numpy as np


DigitRecognizer = Callable[[np.ndarray], int | None]
NameRecognizer = Callable[[np.ndarray], str | None]


@dataclass(frozen=True)
class ScoreboardReading:
    set_number: int | None
    inning: int | None
    player1_score: int | None
    player2_score: int | None
    player1_run: int | None
    player2_run: int | None
    active_color: str | None = None
    row1_color: str | None = None
    player1_name: str | None = None
    player2_name: str | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        result: dict[str, int | str | None] = {
            "set": self.set_number,
            "inning": self.inning,
            "player1Score": self.player1_score,
            "player2Score": self.player2_score,
            "player1Run": self.player1_run,
            "player2Run": self.player2_run,
            "activeColor": self.active_color,
            "row1Color": self.row1_color,
            "player1Name": self.player1_name,
            "player2Name": self.player2_name,
        }
        return result


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


class TesseractDigitRecognizer:
    """Digit-only OCR used by the fa6bfa5 PBA scoreboard pipeline."""

    def __init__(self) -> None:
        self.pytesseract = None
        try:
            import pytesseract

            windows_binary = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            if windows_binary.exists():
                pytesseract.pytesseract.tesseract_cmd = str(windows_binary)
            pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
        except Exception:
            self.pytesseract = None

    @property
    def available(self) -> bool:
        return self.pytesseract is not None

    def __call__(self, image: np.ndarray, modes: tuple[int, ...] = (8, 7, 13)) -> int | None:
        if not self.available or image.size == 0:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, threshold = cv2.threshold(
            up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = cv2.copyMakeBorder(
            threshold, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255
        )
        for mode in modes:
            text = self.pytesseract.image_to_string(
                threshold,
                config=f"--psm {mode} -c tessedit_char_whitelist=0123456789",
            )
            digits = "".join(character for character in text if character.isdigit())
            if digits:
                return int(digits)
        return None


class TesseractNameRecognizer:
    """Korean/English OCR for the two stable player-name rows."""

    def __init__(self) -> None:
        self.pytesseract = None
        self.language = "kor+eng"
        self.tessdata_config = ""
        try:
            import pytesseract

            windows_binary = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            if windows_binary.exists():
                pytesseract.pytesseract.tesseract_cmd = str(windows_binary)
            languages = set(pytesseract.get_languages(config=""))
            if "kor" not in languages:
                local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
                local_tessdata = (
                    Path(local_app_data) / "CueCast" / "tessdata"
                    if local_app_data
                    else None
                )
                if local_tessdata and (local_tessdata / "kor.traineddata").exists():
                    self.language = "kor"
                    self.tessdata_config = f"--tessdata-dir {local_tessdata.as_posix()}"
                    languages = set(
                        pytesseract.get_languages(config=self.tessdata_config)
                    )
            if "kor" in languages:
                self.pytesseract = pytesseract
        except Exception:
            self.pytesseract = None

    @property
    def available(self) -> bool:
        return self.pytesseract is not None

    @staticmethod
    def normalize(text: str) -> str | None:
        value = unicodedata.normalize("NFC", text)
        value = re.sub(r"[^가-힣A-Za-z· ]", "", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value if 2 <= len(value) <= 20 else None

    def __call__(self, image: np.ndarray) -> str | None:
        if not self.available or image.size == 0:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        up = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(
            up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            up,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        # Thin white Korean glyphs sometimes disappear under global Otsu at
        # 360p. Try the original grayscale, Otsu, then adaptive threshold.
        for candidate in (up, otsu, adaptive):
            if float(np.mean(candidate)) < 127:
                candidate = cv2.bitwise_not(candidate)
            candidate = cv2.copyMakeBorder(
                candidate, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255
            )
            text = self.pytesseract.image_to_string(
                candidate,
                lang=self.language,
                config=f"--oem 1 --psm 7 {self.tessdata_config}".strip(),
            )
            normalized = self.normalize(text)
            if normalized is not None:
                return normalized
        return None


class RealtimePbaScoreboardReader:
    """fa6bfa5 score boxes/circles adapted to the live CueCast status contract."""

    YELLOW_LO = np.array([20, 110, 130])
    YELLOW_HI = np.array([38, 255, 255])
    WHITE_S_MAX = 60
    WHITE_V_MIN = 170
    LOCK_FRAMES = 5
    HEARTBEAT_SAMPLES = 20
    NAME_CONFIRMATIONS = 3
    SET_CONFIRMATIONS = 4
    SET_MIN = 1
    SET_MAX = 7
    NAME_CELLS = {
        # Normalized coordinates inside panel_box. The score cells begin at
        # x=0.715, so these crops deliberately stop before them.
        "player1_name": (0.04, 0.29, 0.68, 0.64),
        "player2_name": (0.04, 0.65, 0.68, 0.99),
    }

    def __init__(
        self,
        recognizer: TesseractDigitRecognizer | None = None,
        header_recognizer: DigitRecognizer | None = None,
        name_recognizer: NameRecognizer | None = None,
    ) -> None:
        self.recognizer = recognizer or TesseractDigitRecognizer()
        self.header_recognizer = header_recognizer or SyntheticDigitRecognizer()
        self.name_recognizer = name_recognizer or TesseractNameRecognizer()
        self.enabled = self.recognizer.available
        self.reset()

    def reset(self) -> None:
        self.box_white: tuple[int, int, int, int] | None = None
        self.box_yellow: tuple[int, int, int, int] | None = None
        self.circle_white: tuple[int, int, int, int] | None = None
        self.circle_yellow: tuple[int, int, int, int] | None = None
        self.panel_box: tuple[int, int, int, int] | None = None
        self._box_candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._circle_candidates: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._signature: np.ndarray | None = None
        self._since_ocr = 0
        self._pending: dict[str, tuple[object, int]] = {}
        self._committed: dict[str, object] = {}
        self._names_locked = False

    def reset_scores_and_runs(self) -> None:
        """Forget only score/run OCR decisions while preserving names and layout."""
        keys = (
            "white_score",
            "yellow_score",
            "player1_run",
            "player2_run",
            "active_run_player",
            "run_recheck",
        )
        for key in keys:
            self._committed.pop(key, None)
            self._pending.pop(key, None)
        self._signature = None
        self._since_ocr = self.HEARTBEAT_SAMPLES

    @property
    def locked(self) -> bool:
        return self.box_white is not None and self.box_yellow is not None

    @property
    def circles_locked(self) -> bool:
        return self.circle_white is not None and self.circle_yellow is not None

    @property
    def names_locked(self) -> bool:
        return self._names_locked

    def _try_locate_boxes(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        y0 = int(height * 0.55)
        roi = frame[y0:height, : int(width * 0.5)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.YELLOW_LO, self.YELLOW_HI)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = height * width
        found = None
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not 0.0008 * frame_area < w * h < 0.02 * frame_area:
                continue
            # 저해상도(360p)에서 작은 점수 셀은 검은 숫자가 차지하는 비중이 커
            # 채움비가 0.8 안팎까지 내려간다(예: PBA 챔피언십 24-25 오버레이).
            # 흰 박스 검증·5프레임 안정 lock 이 뒤에 있으므로 0.72 까지 완화.
            if cv2.contourArea(contour) / max(w * h, 1) < 0.72:
                continue
            if not 0.5 < w / max(h, 1) < 2.2:
                continue
            for offset in (-h, h):
                white_y = y + offset
                if white_y < 0 or white_y + h > roi.shape[0]:
                    continue
                white = hsv[white_y : white_y + h, x : x + w]
                fraction = np.mean(
                    (white[..., 1] < self.WHITE_S_MAX)
                    & (white[..., 2] > self.WHITE_V_MIN)
                )
                if fraction > 0.55:
                    found = ((x, white_y + y0, w, h), (x, y + y0, w, h))
                    break
            if found:
                break
        if found is None:
            self._box_candidates.clear()
            return
        self._box_candidates.append(found)
        if len(self._box_candidates) < self.LOCK_FRAMES:
            return
        values = np.asarray(self._box_candidates)
        median = np.median(values, axis=0).astype(int)
        if np.abs(values - median).max() > max(median[0, 2], median[0, 3]):
            self._box_candidates = self._box_candidates[-1:]
            return
        self.box_white = tuple(int(value) for value in median[0])
        self.box_yellow = tuple(int(value) for value in median[1])
        top = min((self.box_white, self.box_yellow), key=lambda box: box[1])
        panel_width = round(top[2] / 0.11)
        panel_height = round(top[3] / 0.35)
        panel_x = round(top[0] - 0.715 * panel_width)
        panel_y = round(top[1] - 0.29 * panel_height)
        self.panel_box = (
            max(0, panel_x),
            max(0, panel_y),
            min(width - max(0, panel_x), panel_width),
            min(height - max(0, panel_y), panel_height),
        )

    def _find_circle(
        self, frame: np.ndarray, box: tuple[int, int, int, int], color: str
    ) -> tuple[int, int, int, int] | None:
        x, y, w, h = box
        x1, x2 = x + w, min(frame.shape[1], x + w + 3 * w)
        y1, y2 = max(0, y - h // 3), min(frame.shape[0], y + h + h // 3)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if color == "white":
            mask = (
                (hsv[..., 1] < self.WHITE_S_MAX)
                & (hsv[..., 2] > self.WHITE_V_MIN)
            ).astype(np.uint8) * 255
        else:
            mask = cv2.inRange(hsv, self.YELLOW_LO, self.YELLOW_HI)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            extent = cv2.contourArea(contour) / max(bw * bh, 1)
            if not 0.5 * h < bh < 1.4 * h or not 0.7 < bw / max(bh, 1) < 1.4:
                continue
            if not 0.6 < extent < 0.92:
                continue
            if best is None or bx < best[0]:
                best = (bx, by, bw, bh)
        if best is None:
            return None
        bx, by, bw, bh = best
        return x1 + bx, y1 + by, bw, bh

    def _try_locate_circles(self, frame: np.ndarray) -> None:
        assert self.box_white is not None and self.box_yellow is not None
        white = self._find_circle(frame, self.box_white, "white")
        yellow = self._find_circle(frame, self.box_yellow, "yellow")
        if white is None or yellow is None:
            self._circle_candidates.clear()
            return
        self._circle_candidates.append((white, yellow))
        if len(self._circle_candidates) < self.LOCK_FRAMES:
            return
        values = np.asarray(self._circle_candidates)
        median = np.median(values, axis=0).astype(int)
        if np.abs(values - median).max() > max(median[0, 2], median[0, 3]):
            self._circle_candidates = self._circle_candidates[-1:]
            return
        self.circle_white = tuple(int(value) for value in median[0])
        self.circle_yellow = tuple(int(value) for value in median[1])
        self._signature = None

    @staticmethod
    def _crop(
        image: np.ndarray, box: tuple[float, float, float, float]
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = box
        return image[
            round(y1 * height) : round(y2 * height),
            round(x1 * width) : round(x2 * width),
        ]

    def _read_colored(
        self, frame: np.ndarray, box: tuple[int, int, int, int], color: str
    ) -> int | None:
        x, y, w, h = box
        crop = frame[y : y + h, x : x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if color == "white":
            fraction = np.mean(
                (hsv[..., 1] < self.WHITE_S_MAX)
                & (hsv[..., 2] > self.WHITE_V_MIN)
            )
        else:
            fraction = np.mean(cv2.inRange(hsv, self.YELLOW_LO, self.YELLOW_HI) > 0)
        if fraction < 0.45:
            return None
        value = self.recognizer(crop)
        if value is None:
            # 360p 저해상도에서 tesseract 가 한 자리 숫자를 통째로 놓치는 경우가
            # 있다(예: 24-25 오버레이의 "0"). 두 자리 점수를 한 자리로 오독하지
            # 않도록 잉크 폭이 셀 폭의 55% 이하일 때만 합성 KNN 으로 폴백한다.
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, ink = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            # 셀 테두리의 어두운 픽셀이 잉크 폭을 왜곡하지 않도록 여백은 제외.
            margin = max(1, round(0.15 * min(ink.shape)))
            interior = ink[margin:-margin, margin:-margin]
            columns = np.where(interior.max(axis=0) > 0)[0]
            if columns.size and (columns[-1] - columns[0] + 1) <= 0.55 * crop.shape[1]:
                value = self.header_recognizer(crop)
        return value if value is not None and value <= 40 else None

    def _read_circle(
        self, frame: np.ndarray, box: tuple[int, int, int, int], color: str
    ) -> tuple[int | None, bool]:
        x, y, w, h = box
        crop = frame[y : y + h, x : x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if color == "white":
            fraction = np.mean(
                (hsv[..., 1] < self.WHITE_S_MAX)
                & (hsv[..., 2] > self.WHITE_V_MIN)
            )
        else:
            fraction = np.mean(cv2.inRange(hsv, self.YELLOW_LO, self.YELLOW_HI) > 0)
        if fraction < 0.35:
            return None, False
        margin = max(2, int(0.18 * min(w, h)))
        inner = cv2.cvtColor(crop[margin : h - margin, margin : w - margin], cv2.COLOR_BGR2GRAY)
        if inner.size == 0 or float(np.mean(inner < 100)) < 0.03:
            return None, False
        value = self.recognizer(inner, (10, 8, 7))
        return (value if value is not None and value <= 30 else None), True

    def _confirm(self, key: str, value: object | None) -> bool:
        if value is None:
            self._pending.pop(key, None)
            return False
        if self._committed.get(key) == value:
            self._pending.pop(key, None)
            return False
        pending = self._pending.get(key)
        count = pending[1] + 1 if pending and pending[0] == value else 1
        if count < 2:
            self._pending[key] = (value, count)
            return False
        self._committed[key] = value
        self._pending.pop(key, None)
        return True

    def _confirm_name(self, key: str, value: object | None) -> bool:
        """Require three identical consecutive OCR reads before publishing a name."""
        if value is None:
            self._pending.pop(key, None)
            return False
        if self._committed.get(key) == value:
            self._pending.pop(key, None)
            return False
        pending = self._pending.get(key)
        count = pending[1] + 1 if pending and pending[0] == value else 1
        if count < self.NAME_CONFIRMATIONS:
            self._pending[key] = (value, count)
            return False
        self._committed[key] = value
        self._pending.pop(key, None)
        return True

    def _confirm_set(self, key: str, value: object | None) -> bool:
        """Use a longer stable window for the tiny 360p set-number glyph."""
        if value is None:
            self._pending.pop(key, None)
            return False
        if self._committed.get(key) == value:
            self._pending.pop(key, None)
            return False
        pending = self._pending.get(key)
        count = pending[1] + 1 if pending and pending[0] == value else 1
        if count < self.SET_CONFIRMATIONS:
            self._pending[key] = (value, count)
            return False
        self._committed[key] = value
        self._pending.pop(key, None)
        return True

    def _confirm_score(self, key: str, value: object | None) -> bool:
        """Publish a non-regressing score immediately after one OCR read."""
        if value is None or self._committed.get(key) == value:
            self._pending.pop(key, None)
            return False
        self._committed[key] = value
        self._pending.pop(key, None)
        return True

    def _committed_reading(self) -> ScoreboardReading:
        assert self.box_white is not None and self.box_yellow is not None
        row1_color = (
            "white" if self.box_white[1] < self.box_yellow[1] else "yellow"
        )
        row2_color = "yellow" if row1_color == "white" else "white"

        def number(key: str) -> int | None:
            value = self._committed.get(key)
            return int(value) if value is not None else None

        return ScoreboardReading(
            set_number=number("set"),
            inning=number("inning"),
            player1_score=number(f"{row1_color}_score"),
            player2_score=number(f"{row2_color}_score"),
            player1_run=number("player1_run"),
            player2_run=number("player2_run"),
            active_color=str(self._committed["active_color"])
            if "active_color" in self._committed
            else None,
            row1_color=row1_color,
            player1_name=str(self._committed["player1_name"])
            if "player1_name" in self._committed
            else None,
            player2_name=str(self._committed["player2_name"])
            if "player2_name" in self._committed
            else None,
        )

    def sample(self, _frame_number: int, frame: np.ndarray) -> ScoreboardReading | None:
        if not self.enabled:
            return None
        if not self.locked:
            self._try_locate_boxes(frame)
            if not self.locked:
                return None
        if not self.circles_locked:
            self._try_locate_circles(frame)

        assert self.box_white is not None and self.box_yellow is not None
        signature_box = self.panel_box or self.box_white
        x, y, w, h = signature_box
        signature = cv2.resize(
            cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY),
            (64, 32),
        )
        self._since_ocr += 1
        changed = self._signature is None or float(
            np.mean(cv2.absdiff(signature, self._signature))
        ) > 3.5
        if not (
            changed
            or self._pending
            or not self._names_locked
            or self._since_ocr >= self.HEARTBEAT_SAMPLES
        ):
            return None
        self._signature = signature
        self._since_ocr = 0

        values: dict[str, object | None] = {
            "white_score": self._read_colored(frame, self.box_white, "white"),
            "yellow_score": self._read_colored(frame, self.box_yellow, "yellow"),
        }
        if self.panel_box is not None:
            px, py, pw, ph = self.panel_box
            panel = frame[py : py + ph, px : px + pw]
            set_number = self.header_recognizer(
                self._crop(panel, PbaScoreboardReader.DIGIT_CELLS["set_number"])
            )
            inning = self.header_recognizer(
                self._crop(panel, PbaScoreboardReader.DIGIT_CELLS["inning"])
            )
            values["set"] = (
                set_number
                if set_number is not None and self.SET_MIN <= set_number <= self.SET_MAX
                else None
            )
            values["inning"] = inning if inning is not None and 0 <= inning <= 99 else None

        active_color = None
        active_run = None
        active_run_player = None
        ambiguous_runs = False
        if self.circles_locked:
            assert self.circle_white is not None and self.circle_yellow is not None
            white_run, white_has_digit = self._read_circle(frame, self.circle_white, "white")
            yellow_run, yellow_has_digit = self._read_circle(frame, self.circle_yellow, "yellow")
            if white_has_digit and not yellow_has_digit:
                active_color, active_run = "white", white_run
                active_run_player = 1 if self.circle_white[1] < self.circle_yellow[1] else 2
            elif yellow_has_digit and not white_has_digit:
                active_color, active_run = "yellow", yellow_run
                active_run_player = 1 if self.circle_yellow[1] < self.circle_white[1] else 2
            elif white_has_digit and yellow_has_digit:
                ambiguous_runs = True
        values["active_color"] = active_color
        if ambiguous_runs:
            self._pending["run_recheck"] = (True, 1)
        else:
            self._pending.pop("run_recheck", None)
        run_cleared = False
        run_changed = False
        if active_run_player is not None:
            if self._committed.get("active_run_player") != active_run_player:
                self._committed["active_run_player"] = active_run_player
                run_changed = True
                for key in ("player1_run", "player2_run"):
                    run_cleared = self._committed.pop(key, None) is not None or run_cleared
                    self._pending.pop(key, None)
            active_run_key = f"player{active_run_player}_run"
            inactive_run_key = f"player{2 if active_run_player == 1 else 1}_run"
            run_cleared = self._committed.pop(inactive_run_key, None) is not None or run_cleared
            self._pending.pop(inactive_run_key, None)
            if active_run is not None and self._committed.get(active_run_key) != active_run:
                self._committed[active_run_key] = active_run
                self._pending.pop(active_run_key, None)
                run_changed = True
        elif ambiguous_runs:
            # Keep the last unambiguous value off-screen until the forced re-read.
            for key in ("player1_run", "player2_run"):
                run_cleared = self._committed.pop(key, None) is not None or run_cleared
                self._pending.pop(key, None)

        current_set = int(self._committed.get("set", -1))
        candidate_set = values.get("set")
        if (
            current_set >= 0
            and candidate_set is not None
            and int(candidate_set) not in (current_set, current_set + 1)
        ):
            values["set"] = None
            candidate_set = None
        set_advances = candidate_set is not None and int(candidate_set) > current_set
        same_set = not set_advances
        if same_set:
            for key in ("white_score", "yellow_score", "inning"):
                current = self._committed.get(key)
                if current is not None and values.get(key) is not None and int(values[key]) < int(current):
                    values[key] = None
            current_inning = self._committed.get("inning")
            if (
                current_inning is not None
                and values.get("inning") is not None
                and int(values["inning"]) > int(current_inning) + 1
            ):
                values["inning"] = None

        committed_changed = run_changed or run_cleared
        for key, value in values.items():
            if key in ("white_score", "yellow_score"):
                confirm = self._confirm_score
            elif key == "set":
                confirm = self._confirm_set
            else:
                confirm = self._confirm
            committed_changed = confirm(key, value) or committed_changed

        # Publish cheap numeric fields before Korean name OCR so names cannot
        # hold back scores, set, inning, run, or active cue color.
        if committed_changed:
            return self._committed_reading()

        names_changed = False
        scores_ready = all(
            key in self._committed for key in ("white_score", "yellow_score")
        )
        if self.panel_box is not None and scores_ready and not self._names_locked:
            px, py, pw, ph = self.panel_box
            panel = frame[py : py + ph, px : px + pw]
            for key, cell in self.NAME_CELLS.items():
                if key in self._committed:
                    continue
                value = self.name_recognizer(self._crop(panel, cell))
                names_changed = self._confirm_name(key, value) or names_changed
        if not self._names_locked and all(
            key in self._committed for key in self.NAME_CELLS
        ):
            self._names_locked = True
            for key in self.NAME_CELLS:
                self._pending.pop(key, None)
        if not names_changed:
            return None
        return self._committed_reading()


class FastPbaCueColorReader(RealtimePbaScoreboardReader):
    """Detect only the active cue-ball color without running Tesseract OCR."""

    CONFIRMATIONS = 2

    def __init__(self) -> None:
        # The inherited locator only needs color/shape state. Avoid constructing
        # digit and Korean OCR recognizers for this latency-sensitive path.
        self.enabled = True
        self.reset()
        self._cue_pending: tuple[str, int] | None = None
        self._cue_current: str | None = None

    def reset(self) -> None:
        super().reset()
        self._cue_pending = None
        self._cue_current = None

    def _has_circle_digit(
        self, frame: np.ndarray, box: tuple[int, int, int, int], color: str
    ) -> bool:
        x, y, w, h = box
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if color == "white":
            fraction = np.mean(
                (hsv[..., 1] < self.WHITE_S_MAX)
                & (hsv[..., 2] > self.WHITE_V_MIN)
            )
        else:
            fraction = np.mean(cv2.inRange(hsv, self.YELLOW_LO, self.YELLOW_HI) > 0)
        if fraction < 0.35:
            return False
        margin = max(2, int(0.18 * min(w, h)))
        inner = cv2.cvtColor(
            crop[margin : h - margin, margin : w - margin], cv2.COLOR_BGR2GRAY
        )
        return bool(inner.size and float(np.mean(inner < 100)) >= 0.03)

    def sample(self, _frame_number: int, frame: np.ndarray) -> str | None:
        if not self.locked:
            self._try_locate_boxes(frame)
            if not self.locked:
                return None
        if not self.circles_locked:
            self._try_locate_circles(frame)
            if not self.circles_locked:
                return None
        assert self.circle_white is not None and self.circle_yellow is not None
        white_has_digit = self._has_circle_digit(frame, self.circle_white, "white")
        yellow_has_digit = self._has_circle_digit(frame, self.circle_yellow, "yellow")
        candidate = (
            "white"
            if white_has_digit and not yellow_has_digit
            else "yellow"
            if yellow_has_digit and not white_has_digit
            else None
        )
        if candidate is None:
            self._cue_pending = None
            return None
        count = (
            self._cue_pending[1] + 1
            if self._cue_pending and self._cue_pending[0] == candidate
            else 1
        )
        self._cue_pending = (candidate, count)
        if count < self.CONFIRMATIONS or candidate == self._cue_current:
            return None
        self._cue_current = candidate
        self._cue_pending = None
        return candidate
