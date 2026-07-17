from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Mapping


BALL_NAMES = ("white_ball", "yellow_ball", "red_ball")
Point = tuple[float, float]


@dataclass(frozen=True)
class PositionSample:
    timestamp: float
    positions: dict[str, Point]


@dataclass(frozen=True)
class StopEvent:
    timestamp: float
    positions: dict[str, Point]


class BallStopDetector:
    """세 공이 안정적으로 정지한 순간에 이벤트를 한 번만 발생시킨다."""

    def __init__(
        self,
        *,
        stable_seconds: float = 0.7,
        stop_threshold: float = 0.002,
        move_threshold: float = 0.005,
    ) -> None:
        if stable_seconds <= 0:
            raise ValueError("stable_seconds는 0보다 커야 합니다.")
        if not 0 < stop_threshold < move_threshold:
            raise ValueError("0 < stop_threshold < move_threshold여야 합니다.")

        self.stable_seconds = stable_seconds
        self.stop_threshold = stop_threshold
        self.move_threshold = move_threshold
        self.samples: deque[PositionSample] = deque()
        self.is_stopped = False
        self.last_emitted_positions: dict[str, Point] | None = None

    def reset(self) -> None:
        self.samples.clear()
        self.is_stopped = False
        self.last_emitted_positions = None

    def update(
        self,
        timestamp: float,
        positions: Mapping[str, Point],
    ) -> StopEvent | None:
        if not all(name in positions for name in BALL_NAMES):
            self.samples.clear()
            return None

        current = {name: tuple(map(float, positions[name])) for name in BALL_NAMES}

        if self.is_stopped and self.last_emitted_positions is not None:
            moved = max(
                self._distance(current[name], self.last_emitted_positions[name])
                for name in BALL_NAMES
            )
            if moved > self.move_threshold:
                self.is_stopped = False
                self.samples.clear()

        self.samples.append(PositionSample(timestamp=timestamp, positions=current))
        cutoff = timestamp - self.stable_seconds
        while len(self.samples) > 1 and self.samples[1].timestamp <= cutoff:
            self.samples.popleft()

        if self.is_stopped or not self._covers_stable_window(timestamp):
            return None
        if not self._positions_are_stable():
            return None

        averaged = self._average_positions()
        self.is_stopped = True
        self.last_emitted_positions = averaged
        return StopEvent(timestamp=timestamp, positions=averaged)

    def _covers_stable_window(self, timestamp: float) -> bool:
        if len(self.samples) < 2:
            return False
        return timestamp - self.samples[0].timestamp >= self.stable_seconds * 0.95

    def _positions_are_stable(self) -> bool:
        for name in BALL_NAMES:
            xs = [sample.positions[name][0] for sample in self.samples]
            ys = [sample.positions[name][1] for sample in self.samples]
            span = hypot(max(xs) - min(xs), max(ys) - min(ys))
            if span > self.stop_threshold:
                return False
        return True

    def _average_positions(self) -> dict[str, Point]:
        count = len(self.samples)
        return {
            name: (
                sum(sample.positions[name][0] for sample in self.samples) / count,
                sum(sample.positions[name][1] for sample in self.samples) / count,
            )
            for name in BALL_NAMES
        }

    @staticmethod
    def _distance(first: Point, second: Point) -> float:
        return hypot(first[0] - second[0], first[1] - second[1])
