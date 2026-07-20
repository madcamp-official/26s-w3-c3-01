from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


BALL_NAMES = ("white_ball", "yellow_ball", "red_ball")


@dataclass(frozen=True)
class BallDetection:
    name: str
    confidence: float
    box: tuple[float, float, float, float]
    center_pixel: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class YOLOBallDetector:
    """Ultralytics YOLO 가중치로 세 종류의 당구공을 검출한다."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.5,
        image_size: int = 1280,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "ultralytics가 설치되지 않았습니다. "
                "`pip install -r requirements.txt`를 실행하세요."
            ) from error

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def detect(self, image: np.ndarray) -> dict[str, BallDetection]:
        kwargs: dict[str, Any] = {
            "source": image,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        result = self.model.predict(**kwargs)[0]
        names = result.names
        best: dict[str, BallDetection] = {}

        if result.boxes is None:
            return best

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = str(names[class_id])
            if name not in BALL_NAMES:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            detection = BallDetection(
                name=name,
                confidence=confidence,
                box=(x1, y1, x2, y2),
                center_pixel=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            )
            if name not in best or confidence > best[name].confidence:
                best[name] = detection

        return best
