from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from cuecast_yolo.detector import BALL_NAMES, YOLOBallDetector
from cuecast_yolo.geometry import TableTransform


COLORS = {
    "white_ball": (255, 255, 255),
    "yellow_ball": (0, 220, 255),
    "red_ball": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이미지에서 당구공 좌표 검출")
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--model", required=True, help="학습된 best.pt 경로")
    parser.add_argument("--table", required=True, help="테이블 모서리 JSON")
    parser.add_argument("--output", default="outputs/image_result.json")
    parser.add_argument("--annotated", default="outputs/image_annotated.jpg")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"이미지를 열 수 없습니다: {args.image}")

    transform = TableTransform.from_json(args.table)
    detector = YOLOBallDetector(
        args.model,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )
    detections = detector.detect(image)

    result: dict[str, object] = {
        "image": str(Path(args.image).resolve()),
        "complete": all(name in detections for name in BALL_NAMES),
        "balls": {},
    }
    balls: dict[str, object] = result["balls"]  # type: ignore[assignment]

    for name, detection in detections.items():
        table_x, table_y = transform.normalize(detection.center_pixel)
        balls[name] = {
            "confidence": detection.confidence,
            "center_pixel": detection.center_pixel,
            "table_position": {"x": table_x, "y": table_y},
            "inside_table": transform.contains(detection.center_pixel),
        }

        x1, y1, x2, y2 = map(int, detection.box)
        color = COLORS[name]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{name} ({table_x:.3f}, {table_y:.3f})"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path = Path(args.output)
    annotated_path = Path(args.annotated)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(annotated_path), image)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
