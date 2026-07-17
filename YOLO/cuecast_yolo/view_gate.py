from __future__ import annotations

import cv2
import numpy as np


def fixed_top_view_ratios(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    ring_width: int = 20,
) -> tuple[float, float]:
    """Return blue-cloth ratios inside and just outside fixed table corners."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (85, 55, 45), (115, 255, 255)) > 0
    polygon = np.round(np.asarray(corners)).astype(np.int32)
    if polygon.shape != (4, 2):
        raise ValueError("table corners must have shape (4, 2)")

    inner_mask = np.zeros(blue.shape, dtype=np.uint8)
    cv2.fillConvexPoly(inner_mask, polygon, 255)
    size = max(3, ring_width * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    expanded = cv2.dilate(inner_mask, kernel) > 0
    inside = inner_mask > 0
    outside_ring = expanded & ~inside

    inner_ratio = float(blue[inside].mean()) if np.any(inside) else 0.0
    outer_ratio = (
        float(blue[outside_ring].mean()) if np.any(outside_ring) else 1.0
    )
    return inner_ratio, outer_ratio


def is_fixed_top_view(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    min_inner_blue_ratio: float = 0.42,
    max_outer_blue_ratio: float = 0.12,
    ring_width: int = 20,
) -> bool:
    inner_ratio, outer_ratio = fixed_top_view_ratios(
        image, corners, ring_width=ring_width
    )
    return (
        inner_ratio >= min_inner_blue_ratio
        and outer_ratio <= max_outer_blue_ratio
    )
