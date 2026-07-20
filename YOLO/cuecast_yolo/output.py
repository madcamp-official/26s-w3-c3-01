from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from .stop_detector import Point, StopEvent


@dataclass(frozen=True)
class CoordinateEvent:
    event_id: int
    timestamp_seconds: float
    frame_number: int
    positions: dict[str, Point]

    @classmethod
    def from_stop_event(
        cls, event_id: int, frame_number: int, event: StopEvent
    ) -> "CoordinateEvent":
        return cls(
            event_id=event_id,
            timestamp_seconds=event.timestamp,
            frame_number=frame_number,
            positions=event.positions,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = format_timestamp(self.timestamp_seconds)
        return data


class EventWriter:
    def __init__(self, jsonl_path: str | Path, csv_path: str | Path) -> None:
        self.jsonl_file: TextIO = Path(jsonl_path).open("w", encoding="utf-8")
        self.csv_file: TextIO = Path(csv_path).open(
            "w", encoding="utf-8-sig", newline=""
        )
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "event_id",
                "timestamp",
                "frame_number",
                "white_x",
                "white_y",
                "yellow_x",
                "yellow_y",
                "red_x",
                "red_y",
            ],
        )
        self.csv_writer.writeheader()

    def write(self, event: CoordinateEvent) -> None:
        self.jsonl_file.write(
            json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        )
        positions = event.positions
        self.csv_writer.writerow(
            {
                "event_id": event.event_id,
                "timestamp": format_timestamp(event.timestamp_seconds),
                "frame_number": event.frame_number,
                "white_x": positions["white_ball"][0],
                "white_y": positions["white_ball"][1],
                "yellow_x": positions["yellow_ball"][0],
                "yellow_y": positions["yellow_ball"][1],
                "red_x": positions["red_ball"][0],
                "red_y": positions["red_ball"][1],
            }
        )
        self.jsonl_file.flush()
        self.csv_file.flush()

    def close(self) -> None:
        self.jsonl_file.close()
        self.csv_file.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def format_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
