from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np


@dataclass(frozen=True)
class _AxisLine:
    coordinate: float
    intervals: tuple[tuple[float, float], ...]


def _edge_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(gray))
    lower = max(20, round(median * 0.45))
    upper = max(lower + 30, min(220, round(median * 1.35)))
    return cv2.Canny(gray, lower, upper)


def _merge_axis_lines(
    values: list[tuple[float, float, float]], *, tolerance: float
) -> list[_AxisLine]:
    """Merge Hough segments that describe the same horizontal/vertical edge."""

    if not values:
        return []
    groups: list[list[tuple[float, float, float]]] = []
    for value in sorted(values, key=lambda item: item[0]):
        if not groups or abs(value[0] - np.mean([item[0] for item in groups[-1]])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)

    result: list[_AxisLine] = []
    for group in groups:
        weights = np.asarray([max(1.0, end - start) for _, start, end in group])
        coordinates = np.asarray([coordinate for coordinate, _, _ in group])
        coordinate = float(np.average(coordinates, weights=weights))
        intervals = tuple((float(start), float(end)) for _, start, end in group)
        result.append(_AxisLine(coordinate, intervals))
    return result


def _interval_coverage(
    intervals: tuple[tuple[float, float], ...], start: float, end: float
) -> float:
    clipped = sorted(
        (max(start, first), min(end, second))
        for first, second in intervals
        if second > start and first < end
    )
    if not clipped or end <= start:
        return 0.0
    covered = 0.0
    current_start, current_end = clipped[0]
    for next_start, next_end in clipped[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            covered += current_end - current_start
            current_start, current_end = next_start, next_end
    covered += current_end - current_start
    return min(1.0, covered / (end - start))


def _interior_cloth_score(
    lab: np.ndarray,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> float:
    """Score whether all four edges open directly onto the same playing cloth."""

    height, width = lab.shape[:2]
    x0, x1 = max(0, round(left)), min(width, round(right))
    y0, y1 = max(0, round(top)), min(height, round(bottom))
    table_width = x1 - x0
    table_height = y1 - y0
    if table_width < 20 or table_height < 10:
        return 0.0

    center = lab[
        y0 + round(table_height * 0.25) : y1 - round(table_height * 0.25),
        x0 + round(table_width * 0.25) : x1 - round(table_width * 0.25),
    ]
    if center.size == 0:
        return 0.0
    representative = np.median(center.reshape(-1, 3), axis=0)
    offset = max(2, round(min(table_width, table_height) * 0.006))
    thickness = max(4, round(min(table_width, table_height) * 0.012))
    margin_x = round(table_width * 0.10)
    margin_y = round(table_height * 0.10)
    bands = (
        lab[y0 + offset : y0 + offset + thickness, x0 + margin_x : x1 - margin_x],
        lab[y1 - offset - thickness : y1 - offset, x0 + margin_x : x1 - margin_x],
        lab[y0 + margin_y : y1 - margin_y, x0 + offset : x0 + offset + thickness],
        lab[y0 + margin_y : y1 - margin_y, x1 - offset - thickness : x1 - offset],
    )
    distances: list[float] = []
    for band in bands:
        if band.size == 0:
            return 0.0
        delta = np.median(band.reshape(-1, 3), axis=0) - representative
        distances.append(
            float(
                np.sqrt(
                    (delta[0] * 0.55) ** 2 + delta[1] ** 2 + delta[2] ** 2
                )
            )
        )
    return float(np.exp(-np.mean(distances) / 24.0))


def detect_inner_table_corners(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.12,
    expected_corners: np.ndarray | None = None,
    max_corner_deviation: float = 18.0,
) -> np.ndarray | None:
    """Find the inner four-cushion boundary without relying on cloth color.

    The supported broadcast view is the same near-overhead view used by the
    coordinate pipeline. Long edge segments are grouped into rectangles and
    the candidate closest to the physical 2:1 carom-table ratio is selected.
    Players or cues may hide part of an edge because segment coverage is
    accumulated from all visible pieces.
    """

    height, width = image.shape[:2]
    if height < 40 or width < 80:
        return None

    scale = min(1.0, 1280.0 / width)
    working = (
        image
        if scale == 1.0
        else cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    )
    work_height, work_width = working.shape[:2]
    edges = _edge_image(working)
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB).astype(np.float32)
    minimum_dimension = min(work_width, work_height)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=max(24, round(minimum_dimension * 0.085)),
        minLineLength=max(35, round(minimum_dimension * 0.15)),
        maxLineGap=max(8, round(minimum_dimension * 0.05)),
    )
    if lines is None:
        return None

    horizontal_segments: list[tuple[float, float, float]] = []
    vertical_segments: list[tuple[float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        if angle <= 8.0 or angle >= 172.0:
            horizontal_segments.append(
                ((float(y1) + float(y2)) / 2, float(min(x1, x2)), float(max(x1, x2)))
            )
        elif 82.0 <= angle <= 98.0:
            vertical_segments.append(
                ((float(x1) + float(x2)) / 2, float(min(y1, y2)), float(max(y1, y2)))
            )

    tolerance = max(2.0, minimum_dimension * 0.008)
    horizontal = _merge_axis_lines(horizontal_segments, tolerance=tolerance)
    vertical = _merge_axis_lines(vertical_segments, tolerance=tolerance)
    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    expected = None
    if expected_corners is not None:
        expected = np.asarray(expected_corners, dtype=np.float32) * scale
        if expected.shape != (4, 2):
            raise ValueError("expected_corners must have shape (4, 2)")

    frame_area = float(work_width * work_height)
    best_score = -1.0
    best_corners: np.ndarray | None = None
    for left, right in combinations(vertical, 2):
        table_width = right.coordinate - left.coordinate
        if table_width < work_width * 0.35 or table_width > work_width * 0.92:
            continue
        for top, bottom in combinations(horizontal, 2):
            table_height = bottom.coordinate - top.coordinate
            if table_height < work_height * 0.22 or table_height > work_height * 0.82:
                continue
            area_ratio = table_width * table_height / frame_area
            if not min_area_ratio <= area_ratio <= 0.72:
                continue
            aspect = table_width / table_height
            if not 1.72 <= aspect <= 2.28:
                continue

            corners = np.float32(
                [
                    [left.coordinate, top.coordinate],
                    [right.coordinate, top.coordinate],
                    [right.coordinate, bottom.coordinate],
                    [left.coordinate, bottom.coordinate],
                ]
            )
            if expected is not None:
                deviations = np.linalg.norm(corners - expected, axis=1)
                if float(deviations.max()) > max_corner_deviation * scale:
                    continue
                expected_score = max(
                    0.0,
                    1.0 - float(deviations.mean()) / (max_corner_deviation * scale),
                )
            else:
                expected_score = 0.0

            side_coverage = np.mean(
                [
                    _interval_coverage(top.intervals, left.coordinate, right.coordinate),
                    _interval_coverage(bottom.intervals, left.coordinate, right.coordinate),
                    _interval_coverage(left.intervals, top.coordinate, bottom.coordinate),
                    _interval_coverage(right.intervals, top.coordinate, bottom.coordinate),
                ]
            )
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 0.28)
            center_x = (left.coordinate + right.coordinate) / 2 / work_width
            center_y = (top.coordinate + bottom.coordinate) / 2 / work_height
            center_score = max(
                0.0,
                1.0 - (abs(center_x - 0.5) + abs(center_y - 0.5)) / 0.75,
            )
            cloth_score = _interior_cloth_score(
                lab,
                left.coordinate,
                top.coordinate,
                right.coordinate,
                bottom.coordinate,
            )
            score = (
                aspect_score * 2.8
                + float(side_coverage) * 2.0
                + center_score * 0.35
                + cloth_score * 2.4
                + expected_score * 2.0
            )
            if score > best_score:
                best_score = score
                best_corners = corners

    if best_corners is None:
        return None

    # Rails usually contribute several nested parallel edges. Starting from
    # the best table-shaped family, walk each side inward to the last long
    # seam. This selects the playing-surface boundary instead of the outer
    # cushion or wooden frame.
    left, top = best_corners[0]
    right, bottom = best_corners[2]
    search_depth = float((bottom - top) * 0.13)
    left_options = [
        line.coordinate
        for line in vertical
        if left <= line.coordinate <= left + search_depth
        and _interval_coverage(line.intervals, top, bottom) >= 0.24
    ]
    right_options = [
        line.coordinate
        for line in vertical
        if right - search_depth <= line.coordinate <= right
        and _interval_coverage(line.intervals, top, bottom) >= 0.24
    ]
    top_options = [
        line.coordinate
        for line in horizontal
        if top <= line.coordinate <= top + search_depth
        and _interval_coverage(line.intervals, left, right) >= 0.24
    ]
    bottom_options = [
        line.coordinate
        for line in horizontal
        if bottom - search_depth <= line.coordinate <= bottom
        and _interval_coverage(line.intervals, left, right) >= 0.24
    ]
    refined_left = max(left_options, default=float(left))
    refined_right = min(right_options, default=float(right))
    refined_top = max(top_options, default=float(top))
    refined_bottom = min(bottom_options, default=float(bottom))
    refined_width = refined_right - refined_left
    refined_height = refined_bottom - refined_top
    if refined_height > 0 and 1.72 <= refined_width / refined_height <= 2.28:
        best_corners = np.float32(
            [
                [refined_left, refined_top],
                [refined_right, refined_top],
                [refined_right, refined_bottom],
                [refined_left, refined_bottom],
            ]
        )
    return best_corners / scale


def edge_support_per_side(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    search_radius: int = 4,
) -> tuple[float, float, float, float]:
    """Measure visible line support for top, right, bottom and left edges."""

    polygon = np.round(np.asarray(corners)).astype(np.int32)
    if polygon.shape != (4, 2):
        raise ValueError("table corners must have shape (4, 2)")
    edges = _edge_image(image)
    size = max(3, search_radius * 2 + 1)
    expanded_edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    )
    scores: list[float] = []
    for index in range(4):
        start = polygon[index].astype(np.float32)
        end = polygon[(index + 1) % 4].astype(np.float32)
        # Corner ornaments are noisy, so score only the middle 90% of each side.
        trimmed_start = np.round(start * 0.95 + end * 0.05).astype(np.int32)
        trimmed_end = np.round(start * 0.05 + end * 0.95).astype(np.int32)
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.line(mask, tuple(trimmed_start), tuple(trimmed_end), 255, 1, cv2.LINE_8)
        line_pixels = mask > 0
        count = int(np.count_nonzero(line_pixels))
        score = (
            float(np.count_nonzero((expanded_edges > 0) & line_pixels) / count)
            if count
            else 0.0
        )
        scores.append(score)
    return tuple(scores)  # type: ignore[return-value]
