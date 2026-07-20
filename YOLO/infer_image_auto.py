from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from cuecast_yolo.color_detector import ColorBallDetector, WARP_HEIGHT, WARP_WIDTH
from cuecast_yolo.detector import BALL_NAMES


COLORS = {
    "white_ball": (255, 255, 255),
    "yellow_ball": (0, 220, 255),
    "red_ball": (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습 모델 없이 파란 당구대 이미지의 공 좌표 추출"
    )
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--output-dir", default="outputs/image_auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"이미지를 열 수 없습니다: {args.image}")

    detector = ColorBallDetector()
    table, detections = detector.detect(image)
    if table is None:
        raise SystemExit("전체 파란 당구대를 찾지 못했습니다.")

    positions = detector.normalized_positions(detections)
    result = {
        "image": str(Path(args.image).resolve()),
        "complete": all(name in detections for name in BALL_NAMES),
        "table_corners_pixel": table.corners.tolist(),
        "coordinate_system": {
            "top_left": [0.0, 0.0],
            "bottom_right": [1.0, 1.0],
        },
        "balls": {
            name: {
                "x": position[0],
                "y": position[1],
                "confidence": detections[name].confidence,
            }
            for name, position in positions.items()
        },
        "missing": [name for name in BALL_NAMES if name not in detections],
    }

    annotated = table.warped.copy()
    for name, (x, y) in positions.items():
        center = (
            round(x * (WARP_WIDTH - 1)),
            round(y * (WARP_HEIGHT - 1)),
        )
        color = COLORS[name]
        cv2.circle(annotated, center, 16, color, 3)
        cv2.putText(
            annotated,
            f"{name} ({x:.4f}, {y:.4f})",
            (center[0] + 18, max(25, center[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "coordinates.json"
    image_path = output_dir / "annotated.jpg"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cv2.imwrite(str(image_path), annotated)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"좌표: {json_path.resolve()}")
    print(f"표시 이미지: {image_path.resolve()}")


if __name__ == "__main__":
    main()
