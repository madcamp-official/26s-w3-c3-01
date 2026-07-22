from __future__ import annotations

from math import hypot
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Callable

import cv2

from .detector import BALL_NAMES
from .precut import PreCutLayoutBuffer
from .scoreboard_reader import (
    FastPbaCueColorReader,
    RealtimePbaScoreboardReader,
    ScoreboardReading,
)
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
        scoreboard_reader: RealtimePbaScoreboardReader | None = None,
        scoreboard_callback: ScoreboardCallback | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.callback = callback
        self.sample_fps = sample_fps
        self.scoreboard_reader = scoreboard_reader or RealtimePbaScoreboardReader()
        self.cue_color_reader = FastPbaCueColorReader()
        self.scoreboard_callback = scoreboard_callback
        self._lock = Lock()
        self._scoreboard_lock = Lock()
        self._cue_color_lock = Lock()
        self._thread: Thread | None = None
        self._stop = Event()
        self._shooter = "white"
        self._shooter_confirmed = False
        self._last_confirmed: dict[str, object] | None = None
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
            self._shooter_confirmed = False
            self._last_confirmed = None
            self._pending_sync_seconds = None
            self._status = {
                "running": True,
                "state": "connecting",
                "source": source,
                "positionSeconds": max(0.0, start_seconds),
                "layouts": 0,
                "completeFrames": 0,
                "scoreboardDetected": False,
                "scoreboardReaderEnabled": self.scoreboard_reader.enabled,
                "scoreboard": None,
                "shooterConfirmed": False,
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
            self._shooter_confirmed = True
            self._status.update(
                shooter=shooter,
                shooterConfirmed=True,
                shooterSource="manual",
            )
        self._republish_last_confirmed()

    def reset_scoreboard(self) -> None:
        """Forget OCR decisions so the current scoreboard can be read again."""
        with self._scoreboard_lock:
            self.scoreboard_reader.reset()
        with self._cue_color_lock:
            self.cue_color_reader.reset()
        with self._lock:
            self._shooter_confirmed = False
            self._status.update(
                scoreboardDetected=False,
                scoreboard=None,
                lastScoreboardSeconds=None,
                shooterConfirmed=False,
                shooterSource=None,
            )

    def _accept_fast_shooter(self, shooter: str, timestamp: float) -> None:
        changed = False
        with self._lock:
            if not self._shooter_confirmed or self._shooter != shooter:
                changed = True
            self._shooter = shooter
            self._shooter_confirmed = True
            self._status.update(
                shooter=shooter,
                shooterConfirmed=True,
                shooterSource="scoreboard_fast",
                lastShooterSeconds=timestamp,
            )
        if changed:
            self._republish_last_confirmed()

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

    def _accept_scoreboard(
        self, reading: ScoreboardReading, timestamp: float
    ) -> None:
        scoreboard = {
            **reading.to_dict(),
            "detectedAtSeconds": timestamp,
        }
        active_color = reading.active_color
        shooter_changed = False
        with self._lock:
            if active_color in ("white", "yellow"):
                if not self._shooter_confirmed or self._shooter != active_color:
                    shooter_changed = True
                self._shooter = active_color
                self._shooter_confirmed = True
            self._status.update(
                scoreboardDetected=True,
                scoreboard=scoreboard,
                lastScoreboardSeconds=timestamp,
                shooter=self._shooter,
                shooterConfirmed=self._shooter_confirmed,
            )
        if self.scoreboard_callback is not None:
            self.scoreboard_callback(scoreboard)
        # 점수판(원형 안 숫자 색)으로 수구가 확정/변경된 순간, 마지막 확정
        # 레이아웃이 있으면 다음 정지를 기다리지 않고 곧바로 확률을 갱신한다.
        if shooter_changed:
            self._republish_last_confirmed()

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
        analysis = {
            "mode": "live",
            "detectedAtSeconds": timestamp,
            "layoutSource": source,
            "detectionConfidence": confidence,
            "ballConfidences": confidences or {},
            "trackingState": state,
            "confirmed": confirmed,
        }
        with self._lock:
            shooter = self._shooter
            shooter_confirmed = self._shooter_confirmed
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
                # 나중에 점수판이 수구를 확정하면 이 레이아웃으로 곧바로
                # 예측을 다시 돌릴 수 있도록 마지막 확정 스냅샷을 보관한다.
                self._last_confirmed = {
                    "positions": dict(positions),
                    "analysis": dict(analysis),
                }
        self.callback(positions, shooter, {**analysis, "shooterConfirmed": shooter_confirmed})

    def _republish_last_confirmed(self) -> None:
        """점수판이 수구를 확정한 순간, 마지막 확정 레이아웃으로 즉시 재예측."""
        with self._lock:
            snapshot = self._last_confirmed
            positions = dict(snapshot["positions"]) if snapshot else None  # type: ignore[index]
            analysis = dict(snapshot["analysis"]) if snapshot else None  # type: ignore[index]
            shooter = self._shooter
            shooter_confirmed = self._shooter_confirmed
        if positions is None or analysis is None:
            return
        analysis["shooterConfirmed"] = shooter_confirmed
        analysis["shooterRefresh"] = True
        self.callback(positions, shooter, analysis)

    def _run(self, source: str, start_seconds: float, stop_event: Event) -> None:
        capture = None
        scoreboard_stop = Event()
        scoreboard_thread: Thread | None = None
        cue_color_thread: Thread | None = None
        try:
            video = self.analyzer.resolve(source)
            capture = cv2.VideoCapture(video.media_url)
            if not capture.isOpened():
                raise RuntimeError("YouTube 분석 스트림을 열 수 없습니다")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            step = max(1, round(fps / max(self.sample_fps, 0.1)))
            frame_number = max(0, round(start_seconds * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            scoreboard_queue: Queue[
                tuple[int, int, float, object]
            ] = Queue(maxsize=1)
            cue_color_queue: Queue[
                tuple[int, int, float, object]
            ] = Queue(maxsize=1)
            scoreboard_epoch = 0

            def scoreboard_loop() -> None:
                active_epoch = -1
                while not scoreboard_stop.is_set():
                    try:
                        epoch, sample_frame, sample_time, sample_image = (
                            scoreboard_queue.get(timeout=0.1)
                        )
                    except Empty:
                        continue
                    with self._scoreboard_lock:
                        if epoch != active_epoch:
                            self.scoreboard_reader.reset()
                            active_epoch = epoch
                        reading = self.scoreboard_reader.sample(
                            sample_frame, sample_image  # type: ignore[arg-type]
                        )
                    if reading is not None:
                        self._accept_scoreboard(reading, sample_time)

            def cue_color_loop() -> None:
                active_epoch = -1
                while not scoreboard_stop.is_set():
                    try:
                        epoch, sample_frame, sample_time, sample_image = (
                            cue_color_queue.get(timeout=0.1)
                        )
                    except Empty:
                        continue
                    with self._cue_color_lock:
                        if epoch != active_epoch:
                            self.cue_color_reader.reset()
                            active_epoch = epoch
                        shooter = self.cue_color_reader.sample(
                            sample_frame, sample_image  # type: ignore[arg-type]
                        )
                    if shooter is not None:
                        self._accept_fast_shooter(shooter, sample_time)

            scoreboard_thread = Thread(
                target=scoreboard_loop,
                name="cuecast-scoreboard-reader",
                daemon=True,
            )
            scoreboard_thread.start()
            cue_color_thread = Thread(
                target=cue_color_loop,
                name="cuecast-fast-cue-color-reader",
                daemon=True,
            )
            cue_color_thread.start()
            # YOLO 중심 좌표의 프레임 간 지터가 테이블 폭의 0.4%(0.004)를 쉽게
            # 넘겨 정지 확정이 거의 안 잡히던 문제를 완화한다. 임계를 1.0%까지
            # 올리고 안정 창을 0.3초로 줄여 확정 빈도와 반응 속도를 높인다.
            stop_detector = BallStopDetector(
                stable_seconds=0.3,
                stop_threshold=0.010,
                move_threshold=0.020,
            )
            precut_buffer = PreCutLayoutBuffer(
                buffer_seconds=0.6,
                sample_count=1,
                max_step=0.02,
                max_span=0.04,
                max_cut_gap=0.2,
            )
            last_scoreboard_sample = -1.0
            previous_top_view = False
            last_published_positions: Layout | None = None
            last_complete_confidences: dict[str, float] = {}
            # A안: 완전 정지 확정을 기다리지 않고 "얼추 정지"한 순간 잠정 발행.
            prev_settle_positions: Layout | None = None
            settle_run = 0
            provisional_emitted = False
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
                    target_frame = max(0, round(sync_seconds * fps))
                    # 유튜브 네트워크 스트림은 읽기 시작 후 POS_FRAMES 중간 seek가 먹지 않는다.
                    # 위치 차이가 크면(사용자 탐색) 스트림을 다시 열어 최초 open+seek 경로를 재현한다.
                    big_jump = abs(target_frame - frame_number) > fps * 2
                    if big_jump:
                        capture.release()
                        capture = cv2.VideoCapture(video.media_url)
                        # 탐색 후엔 카메라 구도가 달라질 수 있으니 테이블 기준도 재획득.
                        self.analyzer.reset_tracking()
                    frame_number = target_frame
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    # 일부 스트림 변종은 프레임 단위 seek 를 조용히 무시한다(0초부터
                    # 재생됨). 실제 도달 위치를 확인하고 어긋나면 msec seek 로 폴백.
                    achieved = float(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
                    if abs(achieved - target_frame) > fps * 2:
                        capture.set(cv2.CAP_PROP_POS_MSEC, float(sync_seconds) * 1000.0)
                        achieved = float(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
                    self._update(
                        lastSeekAchievedSeconds=achieved / fps if fps else None
                    )
                    stop_detector.reset()
                    precut_buffer.clear()
                    scoreboard_epoch += 1
                    last_scoreboard_sample = -1.0
                    previous_top_view = False
                    last_published_positions = None
                    last_complete_confidences = {}
                    prev_settle_positions = None
                    settle_run = 0
                    provisional_emitted = False
                    complete_frames = 0
                    with self._lock:
                        self._shooter_confirmed = False
                        self._last_confirmed = None
                    self._update(
                        syncing=False,
                        lastSyncedSeconds=sync_seconds,
                        positionSeconds=sync_seconds,
                        completeFrames=0,
                        trackingState="syncing",
                        scoreboardDetected=False,
                        scoreboard=None,
                        shooterConfirmed=False,
                    )
                ok, frame = capture.read()
                if not ok:
                    self._update(running=False, state="ended")
                    return
                timestamp = frame_number / fps
                if timestamp - last_scoreboard_sample >= 0.25:
                    last_scoreboard_sample = timestamp
                    item = (scoreboard_epoch, frame_number, timestamp, frame.copy())
                    for sample_queue in (scoreboard_queue, cue_color_queue):
                        try:
                            sample_queue.put_nowait(item)
                        except Full:
                            try:
                                sample_queue.get_nowait()
                            except Empty:
                                pass
                            sample_queue.put_nowait(item)
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
                # A안: 공이 "얼추 정지"하면(연속 2샘플 이동 <= 0.012) 완전 확정을
                # 기다리지 않고 즉시 잠정 배치를 발행해 계산으로 넘긴다. 이후
                # stop_detector 가 정밀 확정하면 평균 좌표로 한 번 더 갱신된다.
                if complete and detected.valid_view:
                    if prev_settle_positions is not None:
                        moved = layout_distance(detected.positions, prev_settle_positions)
                        if moved > 0.020:  # 다시 굴러감 = 새 샷 → 잠정 재무장
                            provisional_emitted = False
                        settle_run = settle_run + 1 if moved <= 0.012 else 0
                    prev_settle_positions = dict(detected.positions)
                    if (
                        stop is None
                        and not provisional_emitted
                        and settle_run >= 2
                        and (
                            last_published_positions is None
                            or layout_distance(
                                detected.positions, last_published_positions
                            ) > 0.05
                        )
                    ):
                        self._publish(
                            detected.positions,
                            timestamp=timestamp,
                            source="settling_provisional",
                            confidence=confidence,
                            confidences=detected.confidences,
                            state="settling",
                            confirmed=True,
                        )
                        last_published_positions = dict(detected.positions)
                        provisional_emitted = True
                else:
                    prev_settle_positions = None
                    settle_run = 0
                    provisional_emitted = False
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
            scoreboard_stop.set()
            if scoreboard_thread is not None:
                scoreboard_thread.join(timeout=1.0)
            if cue_color_thread is not None:
                cue_color_thread.join(timeout=1.0)
            if capture is not None:
                capture.release()
