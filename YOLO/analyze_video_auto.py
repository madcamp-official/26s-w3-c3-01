from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cuecast_yolo.color_detector import ColorBallDetector
from cuecast_yolo.detector import BALL_NAMES
from cuecast_yolo.output import CoordinateEvent, EventWriter, format_timestamp
from cuecast_yolo.stop_detector import BallStopDetector


COLORS = {
    "white_ball": (255, 255, 255),
    "yellow_ball": (0, 220, 255),
    "red_ball": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습 모델 없이 파란 당구대 영상의 정지 배치 자동 추출"
    )
    parser.add_argument("video")
    parser.add_argument(
        "--table",
        help="고정 상단 카메라의 테이블 모서리 JSON. 지정 시 다른 카메라 제외",
    )
    parser.add_argument("--output-dir", default="outputs/auto_video")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--stable-seconds", type=float, default=0.7)
    parser.add_argument("--stop-threshold", type=float, default=0.008)
    parser.add_argument("--move-threshold", type=float, default=0.025)
    parser.add_argument("--duplicate-threshold", type=float, default=0.025)
    parser.add_argument(
        "--min-event-separation",
        type=float,
        default=2.0,
        help="연속 정지 이벤트 사이의 최소 시간(초)",
    )
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def layout_distance(
    first: dict[str, tuple[float, float]],
    second: dict[str, tuple[float, float]],
) -> float:
    return max(
        ((first[name][0] - second[name][0]) ** 2 +
         (first[name][1] - second[name][1]) ** 2) ** 0.5
        for name in BALL_NAMES
    )


def draw_debug(
    warped,
    positions: dict[str, tuple[float, float]],
    event_id: int,
    timestamp: float,
):
    result = warped.copy()
    height, width = result.shape[:2]
    for name, (x, y) in positions.items():
        point = (round(x * (width - 1)), round(y * (height - 1)))
        cv2.circle(result, point, 14, COLORS[name], 3)
        cv2.putText(
            result,
            f"{name} ({x:.3f}, {y:.3f})",
            (point[0] + 16, max(24, point[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLORS[name],
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        result,
        f"event={event_id} time={format_timestamp(timestamp)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, round(source_fps / args.sample_fps))
    expected_corners = None
    if args.table:
        table_config = json.loads(Path(args.table).read_text(encoding="utf-8"))
        expected_corners = np.asarray(table_config["corners"], dtype=np.float32)
    detector = ColorBallDetector(expected_corners=expected_corners)
    stop_detector = BallStopDetector(
        stable_seconds=args.stable_seconds,
        stop_threshold=args.stop_threshold,
        move_threshold=args.move_threshold,
    )

    output_dir = Path(args.output_dir)
    snapshots = output_dir / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    event_id = 0
    analyzed_frames = 0
    full_detection_frames = 0
    unique_layouts: list[dict[str, tuple[float, float]]] = []
    metadata: list[dict] = []
    last_event_timestamp: float | None = None

    try:
        with EventWriter(
            output_dir / "coordinates.jsonl", output_dir / "coordinates.csv"
        ) as writer:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_number % sample_interval != 0:
                    frame_number += 1
                    continue

                timestamp = frame_number / source_fps
                analyzed_frames += 1
                table, detections = detector.detect(frame)
                if table is None or not all(name in detections for name in BALL_NAMES):
                    stop_detector.update(timestamp, {})
                    frame_number += 1
                    continue

                full_detection_frames += 1
                positions = detector.normalized_positions(detections)
                stop_event = stop_detector.update(timestamp, positions)
                if stop_event is not None:
                    too_soon = (
                        last_event_timestamp is not None
                        and timestamp - last_event_timestamp
                        < args.min_event_separation
                    )
                    is_duplicate = any(
                        layout_distance(stop_event.positions, existing)
                        <= args.duplicate_threshold
                        for existing in unique_layouts
                    )
                    if not is_duplicate and not too_soon:
                        event_id += 1
                        last_event_timestamp = timestamp
                        unique_layouts.append(stop_event.positions)
                        event = CoordinateEvent.from_stop_event(
                            event_id, frame_number, stop_event
                        )
                        writer.write(event)
                        debug = draw_debug(
                            table.warped,
                            stop_event.positions,
                            event_id,
                            timestamp,
                        )
                        snapshot_path = snapshots / f"event_{event_id:04d}.jpg"
                        cv2.imwrite(str(snapshot_path), debug)
                        metadata.append(
                            {
                                **event.to_dict(),
                                "snapshot": str(snapshot_path),
                            }
                        )
                        print(
                            f"[{event_id}] {format_timestamp(timestamp)} "
                            f"{stop_event.positions}"
                        )

                if analyzed_frames % 1000 == 0:
                    progress = frame_number / max(total_frames, 1) * 100
                    print(
                        f"진행 {progress:.1f}% | 분석 {analyzed_frames} | "
                        f"완전검출 {full_detection_frames} | 이벤트 {event_id}"
                    )
                frame_number += 1
    finally:
        capture.release()

    summary = {
        "video": str(Path(args.video).resolve()),
        "source_fps": source_fps,
        "sample_fps": args.sample_fps,
        "analyzed_frames": analyzed_frames,
        "full_detection_frames": full_detection_frames,
        "unique_stop_events": event_id,
        "events": metadata,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "events"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
