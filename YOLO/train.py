from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="당구공 YOLO 모델 학습")
    parser.add_argument("--data", default="config/billiard_balls.yaml")
    parser.add_argument("--model", default="weights/yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/billiard_balls")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "ultralytics가 없습니다. `pip install -r requirements.txt`를 실행하세요."
        ) from error

    model = YOLO(args.model)
    options = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": "train",
    }
    if args.device:
        options["device"] = args.device
    model.train(**options)


if __name__ == "__main__":
    main()
