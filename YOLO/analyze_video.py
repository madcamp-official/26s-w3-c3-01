from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cuecast_yolo.detector import BALL_NAMES, YOLOBallDetector
from cuecast_yolo.geometry import TableTransform
from cuecast_yolo.output import CoordinateEvent, EventWriter, format_timestamp
from cuecast_yolo.stop_detector import BallStopDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="영상에서 세 공이 정지할 때마다 좌표 이벤트 생성"
    )
    parser.add_argument("video", help="입력 영상 파일")
    parser.add_argument("--model", required=True, help="학습된 best.pt 경로")
    parser.add_argument("--table", required=True, help="테이블 모서리 JSON")
    parser.add_argument("--output-dir", default="outputs/video")
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--stable-seconds", type=float, default=0.7)
    parser.add_argument("--stop-threshold", type=float, default=0.002)
    parser.add_argument("--move-threshold", type=float, default=0.005)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="각 정지 이벤트 프레임 저장",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps는 0보다 커야 합니다.")

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {args.video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    sample_interval = max(1, round(source_fps / args.sample_fps))

    transform = TableTransform.from_json(args.table)
    detector = YOLOBallDetector(
        args.model,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )
    stop_detector = BallStopDetector(
        stable_seconds=args.stable_seconds,
        stop_threshold=args.stop_threshold,
        move_threshold=args.move_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshots"
    if args.save_snapshots:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    event_id = 0
    try:
        with EventWriter(
            output_dir / "coordinates.jsonl",
            output_dir / "coordinates.csv",
        ) as writer:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_number % sample_interval != 0:
                    frame_number += 1
                    continue

                timestamp = frame_number / source_fps
                detections = detector.detect(frame)
                positions = {
                    name: transform.normalize(detection.center_pixel)
                    for name, detection in detections.items()
                    if transform.contains(detection.center_pixel)
                }

                event = stop_detector.update(timestamp, positions)
                if event is not None:
                    event_id += 1
                    coordinate_event = CoordinateEvent.from_stop_event(
                        event_id, frame_number, event
                    )
                    writer.write(coordinate_event)
                    print(
                        f"[{event_id}] {format_timestamp(timestamp)} "
                        f"white={event.positions['white_ball']} "
                        f"yellow={event.positions['yellow_ball']} "
                        f"red={event.positions['red_ball']}"
                    )
                    if args.save_snapshots:
                        cv2.imwrite(
                            str(snapshot_dir / f"event_{event_id:05d}.jpg"), frame
                        )

                frame_number += 1
    finally:
        capture.release()

    print(f"완료: 정지 이벤트 {event_id}개, 결과={output_dir.resolve()}")


if __name__ == "__main__":
    main()
