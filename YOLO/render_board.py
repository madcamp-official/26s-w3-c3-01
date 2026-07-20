from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


BALL_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (20, 20, 225),
    "yellow": (0, 220, 255),
    "white": (245, 245, 245),
}


def draw_virtual_table(
    coordinates: dict[str, Any],
    output_path: str | Path,
) -> np.ndarray:
    """Render normalized billiard-ball coordinates as a virtual table image.

    Expected coordinate format::

        {
          "red": {"normalized": {"x": 0.25, "y": 0.40}},
          "yellow": {"normalized": {"x": 0.50, "y": 0.55}},
          "white": {"normalized": {"x": 0.75, "y": 0.65}}
        }

    A ball value may be ``None`` when that ball was not detected.
    """
    canvas = render_virtual_table(coordinates)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Failed to write image: {output_path}")

    return canvas


def render_virtual_table(
    coordinates: dict[str, Any],
    *,
    compact_labels: bool = False,
    show_title: bool = True,
) -> np.ndarray:
    """Render a virtual table in memory without writing an image file."""

    width, height = 1400, 800
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)

    outer = (90, 90, 1310, 742)
    rail = 42
    play = (
        outer[0] + rail,
        outer[1] + rail,
        outer[2] - rail,
        outer[3] - rail,
    )

    if show_title:
        cv2.putText(
            canvas,
            "Automatic Table + Ball Coordinate Result",
            (90, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(
        canvas,
        (outer[0], outer[1]),
        (outer[2], outer[3]),
        (105, 150, 190),
        -1,
    )
    cv2.rectangle(
        canvas,
        (outer[0] + 18, outer[1] + 18),
        (outer[2] - 18, outer[3] - 18),
        (65, 95, 125),
        -1,
    )
    cv2.rectangle(
        canvas,
        (play[0], play[1]),
        (play[2], play[3]),
        (210, 125, 30),
        -1,
    )

    for i in range(1, 10):
        x = round(play[0] + (play[2] - play[0]) * i / 10)
        cv2.line(canvas, (x, play[1]), (x, play[3]), (180, 105, 28), 1)
    for i in range(1, 5):
        y = round(play[1] + (play[3] - play[1]) * i / 5)
        cv2.line(canvas, (play[0], y), (play[2], y), (180, 105, 28), 1)

    for name in ("red", "yellow", "white"):
        item = coordinates.get(name)
        if item is None:
            continue

        normalized = item.get("normalized", item)
        nx = float(normalized["x"])
        ny = float(normalized["y"])

        if not 0.0 <= nx <= 1.0 or not 0.0 <= ny <= 1.0:
            raise ValueError(
                f"{name} normalized coordinates must be between 0 and 1: "
                f"x={nx}, y={ny}"
            )

        x = round(play[0] + nx * (play[2] - play[0]))
        y = round(play[1] + ny * (play[3] - play[1]))

        cv2.circle(canvas, (x + 4, y + 5), 22, (25, 25, 25), -1)
        cv2.circle(canvas, (x, y), 19, BALL_COLORS[name], -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 19, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.circle(canvas, (x - 6, y - 7), 5, (255, 255, 255), -1)

        label = (
            name[0].upper()
            if compact_labels
            else f"{name.upper()} ({nx:.4f}, {ny:.4f})"
        )
        label_position = (
            min(x + 28, width - (45 if compact_labels else 320)),
            max(y - 12, 30),
        )
        cv2.putText(
            canvas,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (250, 250, 250),
            1,
            cv2.LINE_AA,
        )

    return canvas
