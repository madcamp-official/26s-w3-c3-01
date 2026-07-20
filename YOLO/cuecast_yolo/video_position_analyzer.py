from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

import cv2
import numpy as np

from .color_detector import ColorBallDetector
from .detector import BALL_NAMES
from .precut import BufferedLayout, PreCutLayoutBuffer
from .view_gate import is_fixed_top_view


@dataclass(frozen=True)
class VideoSource:
    media_url: str
    title: str
    duration_seconds: float | None
    source_kind: str


class VideoPositionAnalyzer:
    """Find the last stable overhead layout before a selected video position."""

    def __init__(
        self,
        *,
        sample_fps: float = 5.0,
        table_config: Path | None = None,
    ) -> None:
        self.sample_fps = sample_fps
        self.detector = ColorBallDetector()
        self._detector_lock = Lock()
        self.reference_corners: np.ndarray | None = None
        self.reference_size = (640.0, 360.0)
        if table_config is not None and table_config.exists():
            payload = json.loads(table_config.read_text(encoding="utf-8"))
            self.reference_corners = np.asarray(payload["corners"], dtype=np.float32)
            self.reference_size = (
                float(payload.get("reference_width", 640.0)),
                float(payload.get("reference_height", 360.0)),
            )
        self._cache: dict[str, tuple[float, VideoSource]] = {}
        self._cache_lock = Lock()

    def resolve(self, source: str) -> VideoSource:
        local_path = Path(source)
        if local_path.exists():
            capture = cv2.VideoCapture(str(local_path.resolve()))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            capture.release()
            duration = frames / fps if fps > 0 and frames > 0 else None
            return VideoSource(
                media_url=str(local_path.resolve()),
                title=local_path.name,
                duration_seconds=duration,
                source_kind="local",
            )

        with self._cache_lock:
            cached = self._cache.get(source)
            if cached and monotonic() - cached[0] < 1800:
                return cached[1]

        try:
            from yt_dlp import YoutubeDL
        except ImportError as error:  # pragma: no cover - dependency error
            raise RuntimeError("yt-dlp가 설치되지 않았습니다") from error

        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": (
                "best[protocol^=http][vcodec!=none][height<=720]/"
                "best[height<=720]/best"
            ),
        }
        try:
            with YoutubeDL(options) as ydl:
                info = ydl.extract_info(source, download=False)
        except Exception as error:
            raise RuntimeError(f"YouTube 영상을 열 수 없습니다: {error}") from error
        if not isinstance(info, dict) or not info.get("url"):
            raise ValueError("YouTube 영상 주소를 가져오지 못했습니다")
        resolved = VideoSource(
            media_url=str(info["url"]),
            title=str(info.get("title") or "YouTube video"),
            duration_seconds=(
                float(info["duration"]) if info.get("duration") is not None else None
            ),
            source_kind="youtube",
        )
        with self._cache_lock:
            self._cache[source] = (monotonic(), resolved)
        return resolved

    def detect_frame(
        self, frame: np.ndarray
    ) -> tuple[dict[str, tuple[float, float]], float] | None:
        """Detect a complete three-ball layout in a calibrated overhead frame."""
        height, width = frame.shape[:2]
        corners = None
        if self.reference_corners is not None:
            corners = self.reference_corners.copy()
            corners[:, 0] *= width / self.reference_size[0]
            corners[:, 1] *= height / self.reference_size[1]

        with self._detector_lock:
            if corners is not None:
                top_view = is_fixed_top_view(
                    frame,
                    corners,
                    min_mean_edge_support=0.50,
                    min_side_edge_support=0.34,
                )
                if not top_view:
                    return None
                table = self.detector.table_view_from_corners(frame, corners)
                detections = self.detector.detect_in_table(table.warped)
            else:
                table, detections = self.detector.detect(frame)
                if table is None:
                    return None
            if not all(name in detections for name in BALL_NAMES):
                return None
            positions = self.detector.normalized_positions(detections)
            confidence = sum(
                float(detections[name].confidence) for name in BALL_NAMES
            ) / len(BALL_NAMES)
        return positions, confidence

    def analyze(
        self,
        source: str,
        selected_seconds: float,
        *,
        lookback_seconds: float = 12.0,
    ) -> dict[str, object]:
        if selected_seconds < 0:
            raise ValueError("영상 위치는 0초 이상이어야 합니다")
        if not 2.0 <= lookback_seconds <= 30.0:
            raise ValueError("검색 구간은 2~30초여야 합니다")

        video = self.resolve(source)
        start_seconds = max(0.0, selected_seconds - lookback_seconds)
        capture = cv2.VideoCapture(video.media_url)
        if not capture.isOpened():
            raise ValueError("분석용 영상 스트림을 열 수 없습니다")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        step = max(1, round(source_fps / max(self.sample_fps, 0.1)))
        start_frame = max(0, round(start_seconds * source_fps))
        end_frame = max(start_frame, round(selected_seconds * source_fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        buffer = PreCutLayoutBuffer(
            buffer_seconds=0.9,
            sample_count=3,
            max_step=0.025,
            max_span=0.05,
            max_cut_gap=0.45,
        )
        previous_complete = False
        candidates: list[tuple[BufferedLayout, float | None, float]] = []
        last_confidence = 0.0
        analyzed_frames = 0
        complete_frames = 0

        try:
            frame_number = start_frame
            while frame_number <= end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                analyzed_frames += 1
                timestamp = frame_number / source_fps
                detected = self.detect_frame(frame)
                complete = detected is not None
                if complete:
                    complete_frames += 1
                    positions, last_confidence = detected
                    buffer.add(timestamp, positions)
                elif previous_complete:
                    layout = buffer.finalize_on_cut(timestamp)
                    if layout is not None:
                        candidates.append((layout, timestamp, last_confidence))
                previous_complete = complete
                for _ in range(step - 1):
                    if not capture.grab():
                        break
                frame_number += step

            if previous_complete:
                layout = buffer.finalize_on_cut(selected_seconds)
                if layout is not None:
                    candidates.append((layout, None, last_confidence))
        finally:
            capture.release()

        if not candidates:
            raise ValueError(
                "선택 위치 앞에서 안정된 상단 테이블의 세 공을 찾지 못했습니다"
            )
        layout, cut_seconds, confidence = max(
            candidates, key=lambda item: item[0].timestamp
        )
        before = {
            color: [
                float(layout.positions[f"{color}_ball"][0]),
                float(layout.positions[f"{color}_ball"][1]),
            ]
            for color in ("white", "yellow", "red")
        }
        return {
            "before": before,
            "detectedAtSeconds": layout.timestamp,
            "cutAtSeconds": cut_seconds,
            "selectedSeconds": selected_seconds,
            "lookbackSeconds": lookback_seconds,
            "detectionConfidence": confidence,
            "analyzedFrames": analyzed_frames,
            "completeFrames": complete_frames,
            "video": {
                "title": video.title,
                "durationSeconds": video.duration_seconds,
                "sourceKind": video.source_kind,
            },
        }
