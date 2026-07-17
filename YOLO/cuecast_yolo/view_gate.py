from __future__ import annotations

import numpy as np

from .table_edges import edge_support_per_side


def fixed_top_view_edge_support(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    search_radius: int = 4,
) -> tuple[float, float, float, float]:
    """Return color-independent support for top, right, bottom and left lines."""

    return edge_support_per_side(
        image, corners, search_radius=search_radius
    )


def is_fixed_top_view(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    min_mean_edge_support: float = 0.58,
    min_side_edge_support: float = 0.42,
    min_supported_sides: int = 3,
    search_radius: int = 4,
) -> bool:
    """Check that the calibrated inner cushion lines are present in the frame."""

    support = fixed_top_view_edge_support(
        image, corners, search_radius=search_radius
    )
    supported_sides = sum(value >= min_side_edge_support for value in support)
    return (
        float(np.mean(support)) >= min_mean_edge_support
        and supported_sides >= min_supported_sides
    )
