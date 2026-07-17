from __future__ import annotations

from dataclasses import dataclass
from math import pi

import cv2
import numpy as np

from .detector import BallDetection


WARP_WIDTH = 1000
WARP_HEIGHT = 500


@dataclass(frozen=True)
class TableView:
    corners: np.ndarray
    warped: np.ndarray
    image_to_table: np.ndarray


class ColorBallDetector:
    """파란 캐롬 테이블 방송 화면을 위한 자동 테이블/공 검출기.

    YOLO 학습 전 영상에서 초기 좌표와 자동 라벨을 생성하기 위한 검출기다.
    전체 테이블이 보이고 장축이 수평에 가까운 화면만 허용한다.
    """

    def __init__(
        self,
        *,
        min_table_area_ratio: float = 0.12,
        max_table_angle: float = 5.0,
        min_frame_margin_ratio: float = 0.02,
        max_opposite_edge_ratio: float = 1.12,
        min_blue_surround_ratio: float = 0.20,
        max_dark_surround_ratio: float = 0.40,
        expected_corners: np.ndarray | None = None,
        max_corner_deviation: float = 15.0,
    ) -> None:
        self.min_table_area_ratio = min_table_area_ratio
        self.max_table_angle = max_table_angle
        self.min_frame_margin_ratio = min_frame_margin_ratio
        self.max_opposite_edge_ratio = max_opposite_edge_ratio
        self.min_blue_surround_ratio = min_blue_surround_ratio
        self.max_dark_surround_ratio = max_dark_surround_ratio
        self.expected_corners = (
            np.asarray(expected_corners, dtype=np.float32)
            if expected_corners is not None
            else None
        )
        self.max_corner_deviation = max_corner_deviation

    def find_table(self, image: np.ndarray) -> TableView | None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (85, 55, 45), (115, 255, 255))
        blue = cv2.morphologyEx(
            blue,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        )
        contours, _ = cv2.findContours(
            blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        frame_area = image.shape[0] * image.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
            contour_area = cv2.contourArea(contour)
            if contour_area < frame_area * self.min_table_area_ratio:
                continue

            rectangle = cv2.minAreaRect(contour)
            corners = self._order_corners(cv2.boxPoints(rectangle))
            if self.expected_corners is None:
                if not self._corners_have_frame_margin(corners, image.shape):
                    continue
                if not self._is_ceiling_geometry(contour):
                    continue
            top_width = np.linalg.norm(corners[1] - corners[0])
            bottom_width = np.linalg.norm(corners[2] - corners[3])
            left_height = np.linalg.norm(corners[3] - corners[0])
            right_height = np.linalg.norm(corners[2] - corners[1])
            width = (top_width + bottom_width) / 2
            height = (left_height + right_height) / 2
            if height <= 0:
                continue
            aspect_ratio = width / height
            if not 1.65 <= aspect_ratio <= 2.25:
                continue

            angle = abs(
                np.degrees(
                    np.arctan2(
                        corners[1][1] - corners[0][1],
                        corners[1][0] - corners[0][0],
                    )
                )
            )
            if angle > self.max_table_angle:
                continue

            rectangle_area = max(width * height, 1.0)
            if contour_area / rectangle_area < 0.70:
                continue

            warp_corners = corners
            if self.expected_corners is not None:
                deviations = np.linalg.norm(
                    corners - self.expected_corners, axis=1
                )
                if float(deviations.max()) > self.max_corner_deviation:
                    continue
                # 고정 카메라는 같은 기준점을 사용해야 좌표 흔들림이 줄어든다.
                warp_corners = self.expected_corners

            destination = np.float32(
                [
                    [0, 0],
                    [WARP_WIDTH - 1, 0],
                    [WARP_WIDTH - 1, WARP_HEIGHT - 1],
                    [0, WARP_HEIGHT - 1],
                ]
            )
            matrix = cv2.getPerspectiveTransform(warp_corners, destination)
            warped = cv2.warpPerspective(image, matrix, (WARP_WIDTH, WARP_HEIGHT))
            return TableView(
                corners=warp_corners, warped=warped, image_to_table=matrix
            )
        return None

    def detect(
        self, image: np.ndarray
    ) -> tuple[TableView | None, dict[str, BallDetection]]:
        table = self.find_table(image)
        if table is None:
            return None, {}

        hsv = cv2.cvtColor(table.warped, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(table.warped)
        blue_background = cv2.inRange(hsv, (85, 55, 45), (115, 255, 255))
        dark_background = cv2.inRange(hsv, (0, 0, 0), (179, 255, 72))
        masks = {
            "white_ball": cv2.inRange(hsv, (0, 0, 135), (179, 82, 255)),
            "yellow_ball": cv2.inRange(hsv, (10, 55, 90), (42, 255, 255)),
            "red_ball": cv2.bitwise_or(
                cv2.inRange(hsv, (0, 80, 55), (9, 255, 255)),
                cv2.inRange(hsv, (145, 75, 55), (179, 255, 255)),
            ),
        }

        # 빨간 공은 빨강 채널이 녹색과 파랑 채널보다 우세해야 한다.
        red_dominance = ((r.astype(np.int16) - g.astype(np.int16) > 25) &
                         (r.astype(np.int16) - b.astype(np.int16) > 20))
        masks["red_ball"] = cv2.bitwise_and(
            masks["red_ball"], (red_dominance.astype(np.uint8) * 255)
        )

        border = 12
        for mask in masks.values():
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0
            cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                dst=mask,
            )

        detections: dict[str, BallDetection] = {}
        for name, mask in masks.items():
            candidates = self._find_candidates(
                mask,
                name,
                blue_background=blue_background,
                dark_background=dark_background,
            )
            if candidates:
                detections[name] = candidates[0]

        # 색상 영역이 큐·선수와 연결되거나 후보의 신뢰도가 낮으면 원 검출로
        # 실제 공 후보를 추가 탐색한다. 주변이 파란 작은 원을 우선한다.
        if len(detections) < len(masks) or any(
            detection.confidence < 0.72 for detection in detections.values()
        ):
            hough_candidates = self._find_hough_candidates(
                table.warped,
                masks,
                blue_background,
                dark_background,
            )
            for name, candidates in hough_candidates.items():
                if not candidates:
                    continue
                if (
                    name not in detections
                    or candidates[0].confidence > detections[name].confidence
                ):
                    detections[name] = candidates[0]
        return table, detections

    def _find_candidates(
        self,
        mask: np.ndarray,
        name: str,
        *,
        blue_background: np.ndarray,
        dark_background: np.ndarray,
    ) -> list[BallDetection]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: list[tuple[float, BallDetection]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 22 <= area <= 900:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if not 5 <= width <= 42 or not 5 <= height <= 42:
                continue
            aspect = width / max(height, 1)
            if not 0.45 <= aspect <= 2.2:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * pi * area / (perimeter * perimeter) if perimeter else 0
            if circularity < 0.22:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
            blue_ratio, dark_ratio = self._surrounding_ratios(
                center_x,
                center_y,
                width,
                height,
                blue_background,
                dark_background,
            )
            if blue_ratio < self.min_blue_surround_ratio:
                continue
            if dark_ratio > self.max_dark_surround_ratio:
                continue

            expected_area = 260.0
            size_score = max(0.0, 1.0 - abs(area - expected_area) / 500.0)
            score = circularity * 0.45 + size_score * 0.20 + blue_ratio * 0.35
            candidates.append(
                (
                    score,
                    BallDetection(
                        name=name,
                        confidence=min(0.99, max(0.01, score)),
                        box=(float(x), float(y), float(x + width), float(y + height)),
                        center_pixel=(float(center_x), float(center_y)),
                    ),
                )
            )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [detection for _, detection in candidates]

    def _find_hough_candidates(
        self,
        image: np.ndarray,
        class_masks: dict[str, np.ndarray],
        blue_background: np.ndarray,
        dark_background: np.ndarray,
    ) -> dict[str, list[BallDetection]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=24,
            param1=80,
            param2=16,
            minRadius=6,
            maxRadius=20,
        )
        result: dict[str, list[BallDetection]] = {
            name: [] for name in class_masks
        }
        if circles is None:
            return result

        height, width = gray.shape
        for center_x, center_y, radius in circles[0]:
            x0 = max(0, int(center_x - radius - 1))
            x1 = min(width, int(center_x + radius + 2))
            y0 = max(0, int(center_y - radius - 1))
            y1 = min(height, int(center_y + radius + 2))
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (radius * 0.8) ** 2
            disk_count = int(np.count_nonzero(disk))
            if disk_count == 0:
                continue

            blue_ratio, dark_ratio = self._surrounding_ratios(
                float(center_x),
                float(center_y),
                round(radius * 2),
                round(radius * 2),
                blue_background,
                dark_background,
            )
            if blue_ratio < self.min_blue_surround_ratio:
                continue
            if dark_ratio > self.max_dark_surround_ratio:
                continue

            radius_score = max(0.0, 1.0 - abs(float(radius) - 10.0) / 12.0)
            for name, mask in class_masks.items():
                class_ratio = float(
                    np.count_nonzero((mask[y0:y1, x0:x1] > 0) & disk)
                    / disk_count
                )
                if class_ratio < 0.16:
                    continue
                score = class_ratio * 0.45 + blue_ratio * 0.35 + radius_score * 0.20
                result[name].append(
                    BallDetection(
                        name=name,
                        confidence=min(0.99, max(0.01, score)),
                        box=(
                            float(center_x - radius),
                            float(center_y - radius),
                            float(center_x + radius),
                            float(center_y + radius),
                        ),
                        center_pixel=(float(center_x), float(center_y)),
                    )
                )

        for candidates in result.values():
            candidates.sort(key=lambda detection: detection.confidence, reverse=True)
        return result

    @staticmethod
    def _surrounding_ratios(
        center_x: float,
        center_y: float,
        width: int,
        height: int,
        blue_background: np.ndarray,
        dark_background: np.ndarray,
    ) -> tuple[float, float]:
        """공 후보 바깥 원형 띠가 파란 천인지 검사한다."""

        radius = max(width, height) / 2.0
        inner_radius = max(5.0, radius * 1.05)
        outer_radius = max(13.0, radius * 2.0)
        x0 = max(0, int(center_x - outer_radius - 1))
        x1 = min(blue_background.shape[1], int(center_x + outer_radius + 2))
        y0 = max(0, int(center_y - outer_radius - 1))
        y1 = min(blue_background.shape[0], int(center_y + outer_radius + 2))
        if x1 <= x0 or y1 <= y0:
            return 0.0, 1.0

        yy, xx = np.ogrid[y0:y1, x0:x1]
        distance_squared = (xx - center_x) ** 2 + (yy - center_y) ** 2
        ring = (
            (distance_squared >= inner_radius**2)
            & (distance_squared <= outer_radius**2)
        )
        ring_count = int(np.count_nonzero(ring))
        if ring_count == 0:
            return 0.0, 1.0

        blue_ratio = float(
            np.count_nonzero((blue_background[y0:y1, x0:x1] > 0) & ring)
            / ring_count
        )
        dark_ratio = float(
            np.count_nonzero((dark_background[y0:y1, x0:x1] > 0) & ring)
            / ring_count
        )
        return blue_ratio, dark_ratio

    def _corners_have_frame_margin(
        self, corners: np.ndarray, image_shape: tuple[int, ...]
    ) -> bool:
        height, width = image_shape[:2]
        margin_x = width * self.min_frame_margin_ratio
        margin_y = height * self.min_frame_margin_ratio
        return bool(
            np.all(corners[:, 0] >= margin_x)
            and np.all(corners[:, 0] <= width - 1 - margin_x)
            and np.all(corners[:, 1] >= margin_y)
            and np.all(corners[:, 1] <= height - 1 - margin_y)
        )

    def _is_ceiling_geometry(self, contour: np.ndarray) -> bool:
        """원근 사다리꼴이 아닌, 직사각형에 가까운 천장뷰만 허용한다."""

        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        if len(polygon) != 4:
            return False
        corners = self._order_corners(polygon.reshape(4, 2))
        top = float(np.linalg.norm(corners[1] - corners[0]))
        bottom = float(np.linalg.norm(corners[2] - corners[3]))
        left = float(np.linalg.norm(corners[3] - corners[0]))
        right = float(np.linalg.norm(corners[2] - corners[1]))
        if min(top, bottom, left, right) <= 0:
            return False

        horizontal_ratio = max(top, bottom) / min(top, bottom)
        vertical_ratio = max(left, right) / min(left, right)
        if horizontal_ratio > self.max_opposite_edge_ratio:
            return False
        if vertical_ratio > self.max_opposite_edge_ratio:
            return False

        for index in range(4):
            previous = corners[(index - 1) % 4] - corners[index]
            following = corners[(index + 1) % 4] - corners[index]
            cosine = float(
                np.dot(previous, following)
                / (np.linalg.norm(previous) * np.linalg.norm(following))
            )
            angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            if not 80.0 <= angle <= 100.0:
                return False
        return True

    @staticmethod
    def normalized_positions(
        detections: dict[str, BallDetection]
    ) -> dict[str, tuple[float, float]]:
        return {
            name: (
                detection.center_pixel[0] / (WARP_WIDTH - 1),
                detection.center_pixel[1] / (WARP_HEIGHT - 1),
            )
            for name, detection in detections.items()
        }

    @staticmethod
    def _order_corners(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        differences = np.diff(points, axis=1).reshape(-1)
        ordered[0] = points[np.argmin(sums)]
        ordered[2] = points[np.argmax(sums)]
        ordered[1] = points[np.argmin(differences)]
        ordered[3] = points[np.argmax(differences)]
        return ordered
