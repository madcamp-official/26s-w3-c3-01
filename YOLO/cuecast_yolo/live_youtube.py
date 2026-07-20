from __future__ import annotations

from math import hypot
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Callable

import cv2

from .detector import BALL_NAMES
from .precut import PreCutLayoutBuffer
from .stop_detector import BallStopDetector
from .video_position_analyzer import VideoPositionAnalyzer


Layout = dict[str, tuple[float, float]]
LayoutCallback = Callable[[Layout, str, dict[str, object]], None]


def layout_distance(first: Layout, second: Layout) -> float:
    return max(
        hypot(
            first[name][0] - second[name][0],
            first[name][1] - second[name][1],
        )
        for name in BALL_NAMES
    )


class YoutubeLiveWorker:
    """Continuously follow a YouTube/VOD stream and publish shot layouts."""

    def __init__(
        self,
        analyzer: VideoPositionAnalyzer,
        callback: LayoutCallback,
        *,
        sample_fps: float = 5.0,
    ) -> None:
        self.analyzer = analyzer
        self.callback = callback
        self.sample_fps = sample_fps
        self._lock = Lock()
        self._thread: Thread | None = None
        self._stop = Event()
        self._shooter = "white"
        self._status: dict[str, object] = {
            "running": False,
            "state": "idle",
            "layouts": 0,
        }

    def start(self, source: str, start_seconds: float, shooter: str) -> None:
        self.stop()
        stop_event = Event()
        with self._lock:
            self._stop = stop_event
            self._shooter = shooter
            self._status = {
                "running": True,
                "state": "connecting",
                "source": source,
                "positionSeconds": max(0.0, start_seconds),
                "layouts": 0,
                "completeFrames": 0,
                "lastError": None,
            }
            self._thread = Thread(
                target=self._run,
                args=(source, max(0.0, start_seconds), stop_event),
                name="cuecast-youtube-live",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            stop_event = self._stop
        stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            if self._thread is thread:
                self._thread = None
                self._status["running"] = False
                if self._status.get("state") not in ("error", "ended"):
                    self._status["state"] = "stopped"

    def set_shooter(self, shooter: str) -> None:
        if shooter not in ("white", "yellow"):
            raise ValueError("shooter는 white 또는 yellow여야 합니다")
        with self._lock:
            self._shooter = shooter

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def _update(self, **values: object) -> None:
        with self._lock:
            self._status.update(values)

    def _publish(
        self,
        positions: Layout,
        *,
        timestamp: float,
        source: str,
        confidence: float,
    ) -> None:
        with self._lock:
            shooter = self._shooter
            layout_count = int(self._status.get("layouts", 0)) + 1
            self._status.update(
                layouts=layout_count,
                lastLayoutSeconds=timestamp,
                lastLayoutSource=source,
            )
        self.callback(
            positions,
            shooter,
            {
                "mode": "live",
                "detectedAtSeconds": timestamp,
                "layoutSource": source,
                "detectionConfidence": confidence,
            },
        )

    def _run(self, source: str, start_seconds: float, stop_event: Event) -> None:
        capture = None
        try:
            video = self.analyzer.resolve(source)
            capture = cv2.VideoCapture(video.media_url)
            if not capture.isOpened():
                raise RuntimeError("YouTube 분석 스트림을 열 수 없습니다")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            step = max(1, round(fps / max(self.sample_fps, 0.1)))
            frame_number = max(0, round(start_seconds * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            stop_detector = BallStopDetector(
                stable_seconds=0.7,
                stop_threshold=0.008,
                move_threshold=0.025,
            )
            precut = PreCutLayoutBuffer(
                buffer_seconds=0.9,
                sample_count=3,
                max_step=0.025,
                max_span=0.05,
                max_cut_gap=0.45,
            )
            previous_complete = False
            last_confidence = 0.0
            last_published: Layout | None = None
            complete_frames = 0
            self._update(
                state="running",
                title=video.title,
                durationSeconds=video.duration_seconds,
            )

            while not stop_event.is_set():
                cycle_started = monotonic()
                ok, frame = capture.read()
                if not ok:
                    self._update(running=False, state="ended")
                    return
                timestamp = frame_number / fps
                detected = self.analyzer.detect_frame(frame)
                positions: Layout = {}
                if detected is not None:
                    positions, last_confidence = detected
                    complete_frames += 1
                    precut.add(timestamp, positions)
                elif previous_complete:
                    layout = precut.finalize_on_cut(timestamp)
                    if layout is not None and (
                        last_published is None
                        or layout_distance(layout.positions, last_published) > 0.012
                    ):
                        self._publish(
                            layout.positions,
                            timestamp=layout.timestamp,
                            source="pre_cut",
                            confidence=last_confidence,
                        )
                        last_published = layout.positions

                stop_layout = stop_detector.update(timestamp, positions)
                if stop_layout is not None and (
                    last_published is None
                    or layout_distance(stop_layout.positions, last_published) > 0.012
                ):
                    self._publish(
                        stop_layout.positions,
                        timestamp=stop_layout.timestamp,
                        source="stopped",
                        confidence=last_confidence,
                    )
                    last_published = stop_layout.positions

                previous_complete = detected is not None
                for _ in range(step - 1):
                    if stop_event.is_set() or not capture.grab():
                        break
                frame_number += step
                self._update(
                    positionSeconds=timestamp,
                    completeFrames=complete_frames,
                )
                remaining = (step / fps) - (monotonic() - cycle_started)
                if remaining > 0:
                    sleep(min(remaining, 0.25))
        except Exception as error:
            self._update(
                running=False,
                state="error",
                lastError=str(error),
            )
        finally:
            if capture is not None:
                capture.release()
