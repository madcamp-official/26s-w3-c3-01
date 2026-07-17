from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class TableTransform:
    """이미지 픽셀 좌표를 당구대 기준 0~1 좌표로 변환한다.

    corners의 순서는 좌상단, 우상단, 우하단, 좌하단이다.
    """

    corners: tuple[Point, Point, Point, Point]

    def __post_init__(self) -> None:
        if len(self.corners) != 4:
            raise ValueError("당구대 모서리는 정확히 4개여야 합니다.")
        src = np.asarray(self.corners, dtype=np.float32)
        if src.shape != (4, 2):
            raise ValueError("각 모서리는 [x, y] 형식이어야 합니다.")

    @classmethod
    def from_json(cls, path: str | Path) -> "TableTransform":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        corners = tuple(tuple(map(float, point)) for point in data["corners"])
        return cls(corners=corners)  # type: ignore[arg-type]

    @property
    def matrix(self) -> np.ndarray:
        source = np.asarray(self.corners, dtype=np.float32)
        destination = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(source, destination)

    def normalize(self, point: Point, *, clamp: bool = False) -> Point:
        source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(source, self.matrix)[0][0]
        x, y = float(transformed[0]), float(transformed[1])
        if clamp:
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
        return x, y

    def normalize_many(self, points: Iterable[Point]) -> list[Point]:
        return [self.normalize(point) for point in points]

    def contains(self, point: Point, *, margin: float = 0.02) -> bool:
        x, y = self.normalize(point)
        return -margin <= x <= 1.0 + margin and -margin <= y <= 1.0 + margin
