from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Mapping

from .detector import BALL_NAMES


Point = tuple[float, float]


@dataclass(frozen=True)
class TrackingUpdate:
    state: str
    positions: dict[str, Point]
    confidences: dict[str, float]
    confirmed_positions: dict[str, Point] | None = None
    cut_positions: dict[str, Point] | None = None


class RealtimeLayoutTracker:
    """Smooth display coordinates while independently confirming still layouts."""

    def __init__(
        self,
        *,
        alpha_moving: float = 0.68,
        alpha_settling: float = 0.35,
        missing_hold_seconds: float = 1.5,
        stable_seconds: float = 0.65,
        settle_threshold: float = 0.006,
        move_threshold: float = 0.012,
        changed_threshold: float = 0.012,
    ) -> None:
        self.alpha_moving = alpha_moving
        self.alpha_settling = alpha_settling
        self.missing_hold_seconds = missing_hold_seconds
        self.stable_seconds = stable_seconds
        self.settle_threshold = settle_threshold
        self.move_threshold = move_threshold
        self.changed_threshold = changed_threshold
        self.filtered: dict[str, Point] = {}
        self.last_seen: dict[str, float] = {}
        self.confidences: dict[str, float] = {}
        self.samples: deque[tuple[float, dict[str, Point]]] = deque()
        self.last_confirmed: dict[str, Point] | None = None
        self.state = "tracking"

    def reset(self) -> None:
        self.filtered.clear()
        self.last_seen.clear()
        self.confidences.clear()
        self.samples.clear()
        self.last_confirmed = None
        self.state = "tracking"

    def update(
        self,
        timestamp: float,
        positions: Mapping[str, Point],
        confidences: Mapping[str, float],
        *,
        valid_view: bool,
    ) -> TrackingUpdate:
        if not valid_view:
            just_left_top_view = self.state != "camera_cut"
            self.samples.clear()
            self.state = "camera_cut"
            recent = {
                name: point
                for name, point in self.filtered.items()
                if timestamp - self.last_seen.get(name, float("-inf"))
                <= self.missing_hold_seconds
            }
            cut_positions = (
                {name: recent[name] for name in BALL_NAMES}
                if just_left_top_view and all(name in recent for name in BALL_NAMES)
                else None
            )
            return TrackingUpdate(
                self.state,
                dict(self.filtered),
                dict(self.confidences),
                cut_positions=cut_positions,
            )

        maximum_step = 0.0
        for name, observed in positions.items():
            if name not in BALL_NAMES:
                continue
            point = (float(observed[0]), float(observed[1]))
            previous = self.filtered.get(name)
            if previous is None:
                filtered = point
            else:
                step = hypot(point[0] - previous[0], point[1] - previous[1])
                maximum_step = max(maximum_step, step)
                alpha = self.alpha_moving if step > self.settle_threshold else self.alpha_settling
                filtered = (
                    previous[0] + alpha * (point[0] - previous[0]),
                    previous[1] + alpha * (point[1] - previous[1]),
                )
            self.filtered[name] = filtered
            self.last_seen[name] = timestamp
            self.confidences[name] = float(confidences.get(name, 0.0))

        visible = {
            name: point
            for name, point in self.filtered.items()
            if timestamp - self.last_seen.get(name, float("-inf"))
            <= self.missing_hold_seconds
        }
        visible_confidence = {name: self.confidences[name] for name in visible}
        if not all(name in visible for name in BALL_NAMES):
            self.samples.clear()
            self.state = "tracking"
            return TrackingUpdate(self.state, visible, visible_confidence)

        complete = {name: visible[name] for name in BALL_NAMES}
        if maximum_step > self.move_threshold:
            self.samples.clear()
            self.state = "moving"
            return TrackingUpdate(self.state, complete, visible_confidence)

        self.samples.append((timestamp, complete))
        cutoff = timestamp - self.stable_seconds
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

        stable = self._covers_window(timestamp) and self._window_span() <= self.settle_threshold
        if not stable:
            self.state = "settling"
            return TrackingUpdate(self.state, complete, visible_confidence)

        confirmed = self._median_layout()
        changed = self.last_confirmed is None or self._layout_distance(
            confirmed, self.last_confirmed
        ) > self.changed_threshold
        self.state = "confirmed"
        if changed:
            self.last_confirmed = confirmed
            return TrackingUpdate(
                self.state, complete, visible_confidence, confirmed_positions=confirmed
            )
        return TrackingUpdate(self.state, complete, visible_confidence)

    def _covers_window(self, timestamp: float) -> bool:
        return len(self.samples) >= 2 and timestamp - self.samples[0][0] >= self.stable_seconds * 0.95

    def _window_span(self) -> float:
        return max(
            hypot(
                max(sample[name][0] for _, sample in self.samples)
                - min(sample[name][0] for _, sample in self.samples),
                max(sample[name][1] for _, sample in self.samples)
                - min(sample[name][1] for _, sample in self.samples),
            )
            for name in BALL_NAMES
        )

    def _median_layout(self) -> dict[str, Point]:
        count = len(self.samples)
        middle = count // 2
        result: dict[str, Point] = {}
        for name in BALL_NAMES:
            xs = sorted(sample[name][0] for _, sample in self.samples)
            ys = sorted(sample[name][1] for _, sample in self.samples)
            if count % 2:
                result[name] = (xs[middle], ys[middle])
            else:
                result[name] = (
                    (xs[middle - 1] + xs[middle]) / 2,
                    (ys[middle - 1] + ys[middle]) / 2,
                )
        return result

    @staticmethod
    def _layout_distance(first: Mapping[str, Point], second: Mapping[str, Point]) -> float:
        return max(hypot(first[n][0] - second[n][0], first[n][1] - second[n][1]) for n in BALL_NAMES)
