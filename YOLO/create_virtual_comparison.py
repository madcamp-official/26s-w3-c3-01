from __future__ import annotations

import argparse
import json
import time
from math import hypot
from pathlib import Path

import cv2
import numpy as np

from cuecast_yolo.color_detector import (
    WARP_HEIGHT,
    WARP_WIDTH,
    ColorBallDetector,
)
from cuecast_yolo.detector import BALL_NAMES, YOLOBallDetector
from cuecast_yolo.output import CoordinateEvent, EventWriter, format_timestamp
from cuecast_yolo.precut import PreCutLayoutBuffer
from cuecast_yolo.stop_detector import BallStopDetector
from cuecast_yolo.view_gate import is_fixed_top_view
from render_board import render_virtual_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a side-by-side video containing the original broadcast and a "
            "virtual table that updates only when all three balls stop."
        )
    )
    parser.add_argument("video", help="Input broadcast video")
    parser.add_argument(
        "--table",
        required=True,
        help="Fixed top-view table corner JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/virtual_comparison",
        help="Output directory",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stable-seconds", type=float, default=0.7)
    parser.add_argument("--stop-threshold", type=float, default=0.008)
    parser.add_argument("--move-threshold", type=float, default=0.025)
    parser.add_argument("--min-blue-ratio", type=float, default=0.42)
    parser.add_argument("--max-outer-blue-ratio", type=float, default=0.12)
    parser.add_argument("--precut-buffer-seconds", type=float, default=0.6)
    parser.add_argument("--precut-max-step", type=float, default=0.02)
    parser.add_argument("--precut-max-span", type=float, default=0.04)
    parser.add_argument("--duplicate-threshold", type=float, default=0.05)
    parser.add_argument(
        "--ball-model",
        help="Optional trained YOLO best.pt with the three billiard-ball classes",
    )
    parser.add_argument("--device")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_corners(path: str) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    corners = np.asarray(payload["corners"], dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("table JSON must contain four [x, y] corners")
    return corners


def to_virtual_coordinates(
    positions: dict[str, tuple[float, float]] | None,
) -> dict[str, dict[str, dict[str, float]]] | dict[str, None]:
    if positions is None:
        return {"white": None, "yellow": None, "red": None}
    return {
        color: {
            "normalized": {
                "x": float(positions[f"{color}_ball"][0]),
                "y": float(positions[f"{color}_ball"][1]),
            }
        }
        for color in ("white", "yellow", "red")
    }


def layout_distance(
    first: dict[str, tuple[float, float]],
    second: dict[str, tuple[float, float]],
) -> float:
    return max(
        hypot(
            first[name][0] - second[name][0],
            first[name][1] - second[name][1],
        )
        for name in BALL_NAMES
    )


def fit_to_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def compose_comparison(
    original: np.ndarray,
    virtual_table: np.ndarray,
    *,
    timestamp: float,
    event_id: int,
    event_timestamp: float | None,
    top_view: bool,
) -> np.ndarray:
    height, width = original.shape[:2]
    left = original.copy()
    right = fit_to_panel(virtual_table, width, height)
    combined = np.hstack((left, right))

    overlay = combined.copy()
    cv2.rectangle(overlay, (0, 0), (combined.shape[1], 34), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.66, combined, 0.34, 0, combined)
    cv2.line(combined, (width, 0), (width, height), (235, 235, 235), 2)

    view_status = "FIXED TOP VIEW" if top_view else "OTHER VIEW"
    cv2.putText(
        combined,
        f"ORIGINAL  {format_timestamp(timestamp)}  {view_status}",
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if event_timestamp is None:
        virtual_status = "VIRTUAL TABLE  waiting for first stopped layout"
    else:
        virtual_status = (
            f"VIRTUAL TABLE  event {event_id}  "
            f"updated {format_timestamp(event_timestamp)}"
        )
    cv2.putText(
        combined,
        virtual_status,
        (width + 12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (90, 255, 90) if event_timestamp is not None else (0, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return combined


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_interval = max(1, round(source_fps / max(args.sample_fps, 0.1)))
    max_frames = (
        round(args.max_seconds * source_fps) if args.max_seconds is not None else None
    )

    output_dir = Path(args.output_dir)
    snapshots_dir = output_dir / "events"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "comparison.mp4"
    video_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width * 2, height),
    )
    if not video_writer.isOpened():
        raise SystemExit(f"cannot create output video: {output_video}")

    corners = load_corners(args.table)
    color_detector = ColorBallDetector()
    yolo_detector = None
    if args.ball_model:
        yolo_detector = YOLOBallDetector(
            args.ball_model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
        )
    stop_detector = BallStopDetector(
        stable_seconds=args.stable_seconds,
        stop_threshold=args.stop_threshold,
        move_threshold=args.move_threshold,
    )
    precut_buffer = PreCutLayoutBuffer(
        buffer_seconds=args.precut_buffer_seconds,
        max_step=args.precut_max_step,
        max_span=args.precut_max_span,
    )

    virtual_frame = render_virtual_table(
        to_virtual_coordinates(None), compact_labels=True, show_title=False
    )
    event_id = 0
    event_timestamp: float | None = None
    frame_number = 0
    top_view = False
    previous_top_view = False
    analyzed_frames = 0
    complete_detection_frames = 0
    accepted_layouts: list[dict[str, tuple[float, float]]] = []
    event_metadata: list[dict[str, object]] = []
    source_counts = {"stable": 0, "pre_cut": 0}
    started = time.perf_counter()

    def accept_layout(
        positions: dict[str, tuple[float, float]],
        *,
        timestamp: float,
        event_frame: int,
        source: str,
        event_writer: EventWriter,
    ) -> bool:
        nonlocal event_id, event_timestamp, virtual_frame
        if any(
            layout_distance(positions, existing) <= args.duplicate_threshold
            for existing in accepted_layouts
        ):
            return False

        event_id += 1
        event_timestamp = timestamp
        accepted_layouts.append(positions)
        event = CoordinateEvent(
            event_id=event_id,
            timestamp_seconds=timestamp,
            frame_number=event_frame,
            positions=positions,
        )
        event_writer.write(event)
        virtual_frame = render_virtual_table(
            to_virtual_coordinates(positions),
            compact_labels=True,
            show_title=False,
        )
        cv2.imwrite(
            str(snapshots_dir / f"event_{event_id:04d}_{source}.png"),
            virtual_frame,
        )
        source_counts[source] += 1
        event_metadata.append({**event.to_dict(), "source": source})
        print(
            f"event {event_id} [{source}]: {format_timestamp(timestamp)} "
            f"{positions}"
        )
        return True

    try:
        with EventWriter(
            output_dir / "events.jsonl", output_dir / "events.csv"
        ) as event_writer:
            while True:
                ok, frame = capture.read()
                if not ok or (max_frames is not None and frame_number >= max_frames):
                    break
                timestamp = frame_number / source_fps

                if frame_number % sample_interval == 0:
                    analyzed_frames += 1
                    previous_top_view = top_view
                    top_view = is_fixed_top_view(
                        frame,
                        corners,
                        min_inner_blue_ratio=args.min_blue_ratio,
                        max_outer_blue_ratio=args.max_outer_blue_ratio,
                    )
                    positions: dict[str, tuple[float, float]] = {}
                    if top_view:
                        table = color_detector.table_view_from_corners(frame, corners)
                        detections = (
                            color_detector.detect_in_table(table.warped)
                            if yolo_detector is None
                            else yolo_detector.detect(table.warped)
                        )
                        if all(name in detections for name in BALL_NAMES):
                            complete_detection_frames += 1
                            positions = color_detector.normalized_positions(detections)
                            precut_buffer.add(timestamp, positions)

                    if previous_top_view and not top_view:
                        pre_cut = precut_buffer.finalize_on_cut(timestamp)
                        if pre_cut is not None:
                            accept_layout(
                                pre_cut.positions,
                                timestamp=pre_cut.timestamp,
                                event_frame=round(pre_cut.timestamp * source_fps),
                                source="pre_cut",
                                event_writer=event_writer,
                            )

                    stop_event = stop_detector.update(timestamp, positions)
                    if stop_event is not None:
                        accept_layout(
                            stop_event.positions,
                            timestamp=stop_event.timestamp,
                            event_frame=frame_number,
                            source="stable",
                            event_writer=event_writer,
                        )

                combined = compose_comparison(
                    frame,
                    virtual_frame,
                    timestamp=timestamp,
                    event_id=event_id,
                    event_timestamp=event_timestamp,
                    top_view=top_view,
                )
                video_writer.write(combined)
                if args.show:
                    cv2.imshow("CueCast stopped-layout comparison", combined)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

                frame_number += 1
                if frame_number % max(1, round(source_fps * 30)) == 0:
                    elapsed = max(time.perf_counter() - started, 0.001)
                    print(
                        f"processed {timestamp / 60:.1f} min | "
                        f"speed {frame_number / elapsed:.1f} FPS | events {event_id}"
                    )
    finally:
        capture.release()
        video_writer.release()
        cv2.destroyAllWindows()

    summary = {
        "video": str(Path(args.video).resolve()),
        "output_video": str(output_video.resolve()),
        "duration_seconds": frame_number / source_fps,
        "source_fps": source_fps,
        "sample_fps": args.sample_fps,
        "analyzed_frames": analyzed_frames,
        "complete_detection_frames": complete_detection_frames,
        "stop_events": event_id,
        "stable_events": source_counts["stable"],
        "pre_cut_events": source_counts["pre_cut"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "event_metadata.json").write_text(
        json.dumps(event_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
