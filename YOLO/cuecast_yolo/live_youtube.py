from __future__ import annotations

from math import hypot
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Callable

import cv2

from .detector import BALL_NAMES
from .precut import PreCutLayoutBuffer
from .scoreboard_reader import PbaScoreboardReader, StableScoreboardState
from .stop_detector import BallStopDetector
from .video_position_analyzer import VideoPositionAnalyzer


Layout = dict[str, tuple[float, float]]
LayoutCallback = Callable[[Layout, str, dict[str, object]], None]
ScoreboardCallback = Callable[[dict[str, object]], None]


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
        sample_fps: float = 10.0,
        scoreboard_reader: PbaScoreboardReader | None = None,
        scoreboard_callback: ScoreboardCallback | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.callback = callback
        self.sample_fps = sample_fps
        self.scoreboard_reader = scoreboard_reader or PbaScoreboardReader()
        self.scoreboard_callback = scoreboard_callback
        self._lock = Lock()
        self._thread: Thread | None = None
        self._stop = Event()
        self._shooter = "white"
        self._pending_sync_seconds: float | None = None
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
            self._pending_sync_seconds = None
            self._status = {
                "running": True,
                "state": "connecting",
                "source": source,
                "positionSeconds": max(0.0, start_seconds),
                "layouts": 0,
                "completeFrames": 0,
                "scoreboardDetected": False,
                "scoreboard": None,
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

    def sync_to(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("동기화 시간은 0 이상이어야 합니다")
        with self._lock:
            if not self._status.get("running"):
                raise ValueError("실시간 분석이 실행 중이 아닙니다")
            self._pending_sync_seconds = float(seconds)
            self._status["syncing"] = True
            self._status["requestedSyncSeconds"] = float(seconds)

    def _take_pending_sync(self) -> float | None:
        with self._lock:
            seconds = self._pending_sync_seconds
            self._pending_sync_seconds = None
            return seconds

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
        state: str,
        confirmed: bool,
        confidences: dict[str, float] | None = None,
    ) -> None:
        with self._lock:
            shooter = self._shooter
            layout_count = int(self._status.get("layouts", 0)) + int(confirmed)
            self._status.update(
                layouts=layout_count,
                trackingState=state,
                lastPreviewSeconds=timestamp,
            )
            if confirmed:
                self._status.update(
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
                "ballConfidences": confidences or {},
                "trackingState": state,
                "confirmed": confirmed,
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
                stable_seconds=0.4,
                stop_threshold=0.004,
                move_threshold=0.015,
            )
            precut_buffer = PreCutLayoutBuffer(
                buffer_seconds=0.6,
                sample_count=1,
                max_step=0.02,
                max_span=0.04,
                max_cut_gap=0.2,
            )
            scoreboard_state = StableScoreboardState(confirmations=3)
            last_scoreboard_sample = -1.0
            previous_top_view = False
            last_published_positions: Layout | None = None
            last_complete_confidences: dict[str, float] = {}
            self.analyzer.reset_tracking()
            complete_frames = 0
            self._update(
                state="running",
                title=video.title,
                durationSeconds=video.duration_seconds,
                detectorMode="minsu_first_top_view_fixed_yolo_stopped_and_precut",
            )

            while not stop_event.is_set():
                cycle_started = monotonic()
                sync_seconds = self._take_pending_sync()
                if sync_seconds is not None:
                    frame_number = max(0, round(sync_seconds * fps))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    stop_detector.reset()
                    precut_buffer.clear()
                    scoreboard_state = StableScoreboardState(confirmations=3)
                    last_scoreboard_sample = -1.0
                    previous_top_view = False
                    last_published_positions = None
                    last_complete_confidences = {}
                    complete_frames = 0
                    self._update(
                        syncing=False,
                        lastSyncedSeconds=sync_seconds,
                        positionSeconds=sync_seconds,
                        completeFrames=0,
                        trackingState="syncing",
                    )
                ok, frame = capture.read()
                if not ok:
                    self._update(running=False, state="ended")
                    return
                timestamp = frame_number / fps
                if timestamp - last_scoreboard_sample >= 0.25:
                    last_scoreboard_sample = timestamp
                    confirmed_scoreboard = scoreboard_state.update(
                        self.scoreboard_reader.read(frame)
                    )
                    if confirmed_scoreboard is not None:
                        scoreboard = {
                            **confirmed_scoreboard.to_dict(),
                            "detectedAtSeconds": timestamp,
                        }
                        self._update(
                            scoreboardDetected=True,
                            scoreboard=scoreboard,
                            lastScoreboardSeconds=timestamp,
                        )
                        if self.scoreboard_callback is not None:
                            self.scoreboard_callback(scoreboard)
                # 5e0cabe pipeline: acquire the table corners once from the first
                # plausible top view, validate that reference on every frame, and
                # use best_3cls.pt YOLO detections for the ball coordinates.
                detected = self.analyzer.detect_tracking_frame(frame)
                complete = all(name in detected.positions for name in BALL_NAMES)
                if complete:
                    complete_frames += 1
                    precut_buffer.add(timestamp, detected.positions)
                    last_complete_confidences = dict(detected.confidences)
                stop = stop_detector.update(
                    timestamp, detected.positions if detected.valid_view else {}
                )
                confidence = (
                    sum(detected.confidences.values()) / len(detected.confidences)
                    if detected.confidences
                    else 0.0
                )
                tracking_state = (
                    "confirmed"
                    if stop is not None
                    else "settling"
                    if complete
                    else "camera_cut"
                    if not detected.valid_view
                    else "missing"
                )
                if previous_top_view and not detected.valid_view:
                    pre_cut = precut_buffer.finalize_on_cut(timestamp)
                    if pre_cut is not None and (
                        last_published_positions is None
                        or layout_distance(
                            pre_cut.positions, last_published_positions
                        ) > 0.05
                    ):
                        pre_cut_confidence = (
                            sum(last_complete_confidences.values())
                            / len(last_complete_confidences)
                            if last_complete_confidences
                            else 0.0
                        )
                        self._publish(
                            pre_cut.positions,
                            timestamp=pre_cut.timestamp,
                            source="pre_cut",
                            confidence=pre_cut_confidence,
                            confidences=last_complete_confidences,
                            state="camera_cut",
                            confirmed=True,
                        )
                        last_published_positions = pre_cut.positions
                if stop is not None:
                    self._publish(
                        stop.positions,
                        timestamp=timestamp,
                        source="stopped",
                        confidence=confidence,
                        confidences=detected.confidences,
                        state="confirmed",
                        confirmed=True,
                    )
                    last_published_positions = stop.positions
                previous_top_view = detected.valid_view
                processing_seconds = monotonic() - cycle_started
                advance = max(step, round(processing_seconds * fps))
                for _ in range(advance - 1):
                    if stop_event.is_set() or not capture.grab():
                        break
                frame_number += advance
                self._update(
                    positionSeconds=timestamp,
                    completeFrames=complete_frames,
                    trackingState=tracking_state,
                )
                remaining = (advance / fps) - (monotonic() - cycle_started)
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
