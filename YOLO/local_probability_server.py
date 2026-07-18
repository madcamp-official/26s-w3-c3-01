from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from cuecast_yolo.shot_probability import (
    BootstrapProbabilityModel,
    CatBoostCoordinateModel,
    HybridShotProbabilityEngine,
    LogisticCoordinateModel,
    layout_from_normalized_colors,
    load_continuous_model,
    load_shot_records,
)
from cuecast_yolo.live_youtube import YoutubeLiveWorker
from cuecast_yolo.video_position_analyzer import VideoPositionAnalyzer


class ProbabilityService:
    def __init__(self, shots_path: Path, model_path: Path) -> None:
        self.records = load_shot_records(shots_path) if shots_path.exists() else []
        if model_path.exists():
            self.model = load_continuous_model(model_path)
        elif not self.records:
            self.model = BootstrapProbabilityModel(0.35)
        elif len(self.records) >= 50 and CatBoostCoordinateModel.is_available():
            self.model = CatBoostCoordinateModel.fit(self.records)
        else:
            self.model = LogisticCoordinateModel.fit(self.records)
        self.engine = HybridShotProbabilityEngine(self.records, self.model)

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "records": len(self.records),
            "modelVersion": self.model.version,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, object]:
        shooter = str(payload["shooter"])
        layout, roles = layout_from_normalized_colors(payload["before"], shooter)
        result = self.engine.predict(
            layout,
            shooter,
            position_error_mm=float(payload.get("position_error_mm", 25.0)),
            prediction_id=payload.get("prediction_id"),
        )
        result["roles"] = {
            "cue": roles[0],
            "object1": roles[1],
            "object2": roles[2],
        }
        result["dataRecords"] = len(self.records)
        return result


class DetectionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._version = 0
        self._value: dict[str, object] | None = None

    def put(self, value: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._version += 1
            self._value = {"version": self._version, **value}
            return dict(self._value)

    def get(self) -> dict[str, object]:
        with self._lock:
            return dict(self._value or {"version": self._version})


def create_handler(
    service: ProbabilityService,
    ui_path: Path,
    detections: DetectionStore,
    video_analyzer: VideoPositionAnalyzer,
    live_worker: YoutubeLiveWorker,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Invalid request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/v1/health":
                self._send_json(service.health())
                return
            if path == "/api/v1/detection/latest":
                self._send_json(detections.get())
                return
            if path == "/api/v1/youtube/live/status":
                self._send_json(live_worker.status())
                return
            if path in ("/", "/index.html"):
                body = ui_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
                if path == "/api/v1/shot-probability":
                    self._send_json(service.predict(payload))
                    return
                if path == "/api/v1/detection":
                    prediction = service.predict(payload)
                    stored = detections.put(
                        {
                            "before": payload["before"],
                            "shooter": payload["shooter"],
                            "prediction": prediction,
                        }
                    )
                    self._send_json(stored, HTTPStatus.CREATED)
                    return
                if path == "/api/v1/youtube/info":
                    video = video_analyzer.resolve(str(payload["url"]))
                    self._send_json(
                        {
                            "title": video.title,
                            "durationSeconds": video.duration_seconds,
                            "sourceKind": video.source_kind,
                        }
                    )
                    return
                if path == "/api/v1/youtube/analyze":
                    analysis = video_analyzer.analyze(
                        str(payload["url"]),
                        float(payload["timestamp_seconds"]),
                        lookback_seconds=float(payload.get("lookback_seconds", 12.0)),
                    )
                    prediction = service.predict(
                        {
                            "before": analysis["before"],
                            "shooter": payload.get("shooter", "white"),
                            "position_error_mm": payload.get(
                                "position_error_mm", 25.0
                            ),
                        }
                    )
                    stored = detections.put(
                        {
                            "before": analysis["before"],
                            "shooter": payload.get("shooter", "white"),
                            "prediction": prediction,
                            "analysis": analysis,
                        }
                    )
                    self._send_json(stored, HTTPStatus.CREATED)
                    return
                if path == "/api/v1/youtube/live/start":
                    shooter = str(payload.get("shooter", "white"))
                    if shooter not in ("white", "yellow"):
                        raise ValueError("shooter는 white 또는 yellow여야 합니다")
                    live_worker.start(
                        str(payload["url"]),
                        float(payload.get("timestamp_seconds", 0.0)),
                        shooter,
                    )
                    self._send_json(live_worker.status(), HTTPStatus.ACCEPTED)
                    return
                if path == "/api/v1/youtube/live/stop":
                    live_worker.stop()
                    self._send_json(live_worker.status())
                    return
                if path == "/api/v1/youtube/live/shooter":
                    live_worker.set_shooter(str(payload["shooter"]))
                    self._send_json(live_worker.status())
                    return
                self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(
                    {"error": "invalid_request", "detail": str(error)},
                    HTTPStatus.BAD_REQUEST,
                )
            except RuntimeError as error:
                self._send_json(
                    {"error": "analysis_failed", "detail": str(error)},
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )

        def log_message(self, format: str, *args: object) -> None:
            print(f"[local-ui] {self.address_string()} {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Serve the local CueCast UI and engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--shots", type=Path, default=root.parent / "data")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "outputs" / "probability_db_catboost" / "model.json",
    )
    parser.add_argument("--ui", type=Path, default=root / "ui" / "index.html")
    parser.add_argument(
        "--table",
        type=Path,
        default=root / "config" / "video1_table.json",
        help="Fixed overhead table calibration used by the broadcast detector",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ProbabilityService(args.shots.resolve(), args.model.resolve())
    detections = DetectionStore()
    video_analyzer = VideoPositionAnalyzer(table_config=args.table.resolve())

    def publish_live_layout(
        positions: dict[str, tuple[float, float]],
        shooter: str,
        analysis: dict[str, object],
    ) -> None:
        before = {
            color: list(positions[f"{color}_ball"])
            for color in ("white", "yellow", "red")
        }
        prediction = service.predict(
            {
                "before": before,
                "shooter": shooter,
                "position_error_mm": 25.0,
            }
        )
        detections.put(
            {
                "before": before,
                "shooter": shooter,
                "prediction": prediction,
                "analysis": analysis,
            }
        )

    live_worker = YoutubeLiveWorker(video_analyzer, publish_live_layout)
    handler = create_handler(
        service,
        args.ui.resolve(),
        detections,
        video_analyzer,
        live_worker,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"CueCast local UI: http://{args.host}:{args.port} | "
        f"records={len(service.records)} | model={service.model.version}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        live_worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
