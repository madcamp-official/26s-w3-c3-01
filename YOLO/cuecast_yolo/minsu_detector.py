from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import BALL_NAMES, YOLOBallDetector
from .table_edges import detect_inner_table_corners


CLOTH_HSV_RANGES = (
    ((90, 80, 80), (130, 255, 255)),
    ((65, 60, 40), (90, 255, 255)),
    ((90, 8, 110), (125, 60, 190)),
    # 경기장 배경·바닥까지 파란 계열인 중계(예: PBA 챔피언십 24-25)에서는 위
    # 넓은 범위가 화면 대부분을 덮어 테이블 분리가 안 된다. 밝은 천만 잡는
    # 고채도·고명도 범위를 추가해 배경(V~150)·바닥(V~90)과 분리한다.
    ((95, 130, 170), (114, 255, 255)),
)


@dataclass(frozen=True)
class TrackingFrame:
    positions: dict[str, tuple[float, float]]
    confidences: dict[str, float]
    valid_view: bool


def _ordered_quad(contour: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(contour)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(contour))
    points = np.asarray(quad).reshape(-1, 2).astype(np.float32)
    sums = points.sum(axis=1)
    differences = points[:, 1] - points[:, 0]
    return np.float32(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ]
    )


def plausible_top_view(corners: np.ndarray | None, shape: tuple[int, ...]) -> bool:
    if corners is None:
        return False
    height, width = shape[:2]
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    if min(top, bottom, left, right) < 1:
        return False
    aspect = (top + bottom) / (left + right)
    horizontal = (
        abs(corners[1, 1] - corners[0, 1]) < 0.05 * height
        and abs(corners[2, 1] - corners[3, 1]) < 0.05 * height
    )
    area = cv2.contourArea(corners) / float(width * height)
    margin_x, margin_y = 0.01 * width, 0.01 * height
    inside = (
        corners[:, 0].min() > margin_x
        and corners[:, 0].max() < width - margin_x
        and corners[:, 1].min() > margin_y
        and corners[:, 1].max() < height - margin_y
    )
    return 1.6 < aspect < 2.4 and horizontal and 0.12 < area < 0.8 and inside


def find_table_corners(image: np.ndarray, close_kernel: int = 15) -> np.ndarray:
    """The cloth-mask table detector used by origin/temp/minsu."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    best: np.ndarray | None = None
    best_area = 0.0
    for lower, upper in CLOTH_HSV_RANGES:
        mask = cv2.inRange(hsv, np.asarray(lower), np.asarray(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        quad = _ordered_quad(max(contours, key=cv2.contourArea))
        area = cv2.contourArea(quad)
        if plausible_top_view(quad, image.shape) and area > best_area:
            best, best_area = quad, area
    if best is None:
        raise ValueError("top-view billiard table was not found")
    return best


def detect_corners_fast(frame: np.ndarray, scale: float = 0.25) -> np.ndarray | None:
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    try:
        return find_table_corners(
            small, close_kernel=max(3, round(15 * scale))
        ) / scale
    except (ValueError, cv2.error):
        return None


def pixels_to_table(points: list[tuple[float, float]], corners: np.ndarray) -> np.ndarray:
    destination = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    matrix = cv2.getPerspectiveTransform(corners, destination)
    values = np.float32(points).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(values, matrix).reshape(-1, 2)


def valid_inner_corners(
    inner: np.ndarray | None,
    view: np.ndarray,
    shape: tuple[int, ...],
) -> bool:
    """Reject an inner-boundary candidate that is not contained by the table view."""

    if inner is None or not plausible_top_view(inner, shape):
        return False
    view_area = float(cv2.contourArea(view))
    inner_area = float(cv2.contourArea(inner))
    if view_area <= 0 or not 0.55 <= inner_area / view_area <= 1.01:
        return False

    height, width = shape[:2]
    tolerance_x, tolerance_y = width * 0.015, height * 0.015
    return bool(
        inner[:, 0].min() >= view[:, 0].min() - tolerance_x
        and inner[:, 0].max() <= view[:, 0].max() + tolerance_x
        and inner[:, 1].min() >= view[:, 1].min() - tolerance_y
        and inner[:, 1].max() <= view[:, 1].max() + tolerance_y
    )


class MinsuRealtimeDetector:
    """Minsu YOLO detector plus its per-frame top-view/camera-cut gate."""

    def __init__(self, model_path: Path, *, confidence: float = 0.25) -> None:
        self.detector = YOLOBallDetector(
            model_path, confidence=confidence, image_size=640
        )
        # The cloth segmentation boundary is retained only for camera-view
        # validation. Ball coordinates are normalized against the separately
        # detected inner cushion seam (the physical 2844 x 1422 mm surface).
        self.view_corners: np.ndarray | None = None
        self.reference_corners: np.ndarray | None = None

    def reset(self) -> None:
        self.view_corners = None
        self.reference_corners = None

    def detect(self, frame: np.ndarray) -> TrackingFrame:
        fast = detect_corners_fast(frame)
        if self.reference_corners is None and plausible_top_view(fast, frame.shape):
            try:
                view_corners = find_table_corners(frame)
                inner_corners = detect_inner_table_corners(frame)
                if valid_inner_corners(inner_corners, view_corners, frame.shape):
                    self.view_corners = view_corners
                    self.reference_corners = inner_corners
            except (ValueError, cv2.error):
                pass

        valid_view = bool(
            self.view_corners is not None
            and self.reference_corners is not None
            and plausible_top_view(fast, frame.shape)
            and float(np.abs(fast - self.view_corners).max()) < 60.0
        )
        if not valid_view or self.reference_corners is None:
            return TrackingFrame({}, {}, False)

        detections = self.detector.detect(frame)
        if not detections:
            return TrackingFrame({}, {}, True)
        names = list(detections)
        transformed = pixels_to_table(
            [detections[name].center_pixel for name in names], self.reference_corners
        )
        positions: dict[str, tuple[float, float]] = {}
        confidences: dict[str, float] = {}
        for name, (x, y) in zip(names, transformed):
            if -0.03 <= x <= 1.03 and -0.03 <= y <= 1.03:
                positions[name] = (float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1)))
                confidences[name] = detections[name].confidence
        return TrackingFrame(positions, confidences, True)

    @property
    def complete_names(self) -> tuple[str, ...]:
        return BALL_NAMES
