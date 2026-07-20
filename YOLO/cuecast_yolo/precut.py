from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import Mapping

from .detector import BALL_NAMES


Point = tuple[float, float]


@dataclass(frozen=True)
class BufferedLayout:
    timestamp: float
    positions: dict[str, Point]


class PreCutLayoutBuffer:
    """Finalize a nearly stopped layout from frames immediately before a cut."""

    def __init__(
        self,
        *,
        buffer_seconds: float = 0.6,
        sample_count: int = 3,
        max_step: float = 0.02,
        max_span: float = 0.04,
        max_cut_gap: float = 0.2,
    ) -> None:
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be positive")
        if sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        self.buffer_seconds = buffer_seconds
        self.sample_count = sample_count
        self.max_step = max_step
        self.max_span = max_span
        self.max_cut_gap = max_cut_gap
        self.samples: deque[BufferedLayout] = deque()

    def add(self, timestamp: float, positions: Mapping[str, Point]) -> None:
        if not all(name in positions for name in BALL_NAMES):
            return
        current = {
            name: tuple(map(float, positions[name])) for name in BALL_NAMES
        }
        self.samples.append(BufferedLayout(timestamp, current))
        cutoff = timestamp - self.buffer_seconds
        while self.samples and self.samples[0].timestamp < cutoff:
            self.samples.popleft()

    def finalize_on_cut(self, cut_timestamp: float) -> BufferedLayout | None:
        if len(self.samples) < self.sample_count:
            self.samples.clear()
            return None

        selected = list(self.samples)[-self.sample_count :]
        self.samples.clear()
        if cut_timestamp - selected[-1].timestamp > self.max_cut_gap:
            return None

        if len(selected) >= 2:
            steps = [
                self._layout_distance(first.positions, second.positions)
                for first, second in zip(selected, selected[1:])
            ]
            span = self._layout_distance(
                selected[0].positions, selected[-1].positions
            )
            if max(steps) > self.max_step or span > self.max_span:
                return None

            # Allow small jitter, but reject a clear acceleration immediately before the cut.
            if len(steps) >= 2 and steps[-1] > steps[-2] * 1.35 + 0.003:
                return None

        positions = {
            name: (
                float(median(sample.positions[name][0] for sample in selected)),
                float(median(sample.positions[name][1] for sample in selected)),
            )
            for name in BALL_NAMES
        }
        return BufferedLayout(selected[-1].timestamp, positions)

    def clear(self) -> None:
        self.samples.clear()

    @staticmethod
    def _layout_distance(
        first: Mapping[str, Point], second: Mapping[str, Point]
    ) -> float:
        return max(
            hypot(
                first[name][0] - second[name][0],
                first[name][1] - second[name][1],
            )
            for name in BALL_NAMES
        )
