from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv

from cuecast_yolo.shot_probability import (
    BootstrapProbabilityModel,
    CatBoostCoordinateModel,
    HybridShotProbabilityEngine,
    LogisticCoordinateModel,
    layout_from_normalized_colors,
    load_continuous_model,
    load_shot_records,
)
from cuecast_yolo.symmetric_probability import (
    DEFAULT_SYMMETRIC_GRID,
    SymmetricCatBoostCoordinateModel,
    SymmetricHybridShotProbabilityEngine,
    load_calibration,
)
from cuecast_yolo.live_youtube import YoutubeLiveWorker
from cuecast_yolo.live_match_state import LiveMatchCoordinator
from cuecast_yolo.prematch_probability import (
    PrematchDataError,
    PrematchService,
    create_prematch_service,
)
from cuecast_yolo.prematch_live_inputs import (
    LockedScoreboardPlayerMatcher,
    PrematchLiveInputProvider,
)
from cuecast_yolo.video_position_analyzer import VideoPositionAnalyzer


class ProbabilityService:
    def __init__(
        self,
        shots_path: Path,
        model_path: Path,
        calibration_path: Path | None = None,
    ) -> None:
        loaded_records = load_shot_records(shots_path) if shots_path.exists() else []
        self.records = [
            record for record in loaded_records if record.cue_ball in ("white", "yellow")
        ]
        if (
            model_path.exists()
            and calibration_path is not None
            and calibration_path.exists()
            and "symmetric_hybrid_v2" in str(model_path)
        ):
            self.model = SymmetricCatBoostCoordinateModel.load(model_path)
            self.engine = SymmetricHybridShotProbabilityEngine(
                self.records,
                self.model,
                grid_config=DEFAULT_SYMMETRIC_GRID,
                calibration=load_calibration(calibration_path),
            )
        elif model_path.exists():
            self.model = load_continuous_model(model_path)
            self.engine = HybridShotProbabilityEngine(self.records, self.model)
        elif not self.records:
            self.model = BootstrapProbabilityModel(0.35)
            self.engine = HybridShotProbabilityEngine(self.records, self.model)
        elif len(self.records) >= 50 and CatBoostCoordinateModel.is_available():
            self.model = CatBoostCoordinateModel.fit(self.records)
            self.engine = HybridShotProbabilityEngine(self.records, self.model)
        else:
            self.model = LogisticCoordinateModel.fit(self.records)
            self.engine = HybridShotProbabilityEngine(self.records, self.model)

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "records": len(self.records),
            "modelVersion": self.model.version,
            "engineVersion": (
                "symmetric-hybrid-v2"
                if isinstance(self.engine, SymmetricHybridShotProbabilityEngine)
                else "hybrid-v1"
            ),
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
        self._confirmed_version = 0
        self._confirmed: dict[str, object] = {}
        self._scoreboard: dict[str, object] = {}

    def put(self, value: dict[str, object]) -> dict[str, object]:
        with self._lock:
            self._version += 1
            if value.get("prediction") is not None:
                self._confirmed_version += 1
                self._confirmed = {
                    "confirmedVersion": self._confirmed_version,
                    "confirmedBefore": value.get("before"),
                    "confirmedPrediction": value.get("prediction"),
                    "confirmedShooter": value.get("shooter"),
                    "confirmedAnalysis": value.get("analysis"),
                }
            self._value = {"version": self._version, **value}
            return {**self._value, **self._confirmed, **self._scoreboard}

    def put_scoreboard(self, scoreboard: dict[str, object]) -> dict[str, object]:
        with self._lock:
            previous = self._scoreboard.get("scoreboard")
            merged = {
                **(previous if isinstance(previous, dict) else {}),
                **{
                    key: value
                    for key, value in scoreboard.items()
                    if value is not None or key in ("player1Run", "player2Run")
                },
            }
            self._scoreboard = {"scoreboard": merged}
            return dict(self._scoreboard)

    def clear_scoreboard(self) -> dict[str, object]:
        with self._lock:
            self._scoreboard = {}
            return {"scoreboard": None}

    def get(self) -> dict[str, object]:
        with self._lock:
            return {
                **(self._value or {"version": self._version}),
                **self._confirmed,
                **self._scoreboard,
            }

    def clear(self) -> dict[str, object]:
        with self._lock:
            self._version += 1
            self._confirmed_version = 0
            self._confirmed = {}
            self._scoreboard = {}
            self._value = {"version": self._version, "pending": True}
            return dict(self._value)


def create_handler(
    service: ProbabilityService,
    prematch_service: PrematchService,
    ui_path: Path,
    extension_dir: Path,
    detections: DetectionStore,
    video_analyzer: VideoPositionAnalyzer,
    live_worker: YoutubeLiveWorker,
    live_match: LiveMatchCoordinator,
    scoreboard_names: LockedScoreboardPlayerMatcher,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
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

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/v1/health":
                self._send_json({**service.health(), "prematchSource": prematch_service.source})
                return
            if path == "/api/v1/prematch/players":
                try:
                    league = query.get("league", ["PBA"])[0]
                    active_only = query.get("active_only", ["true"])[0].casefold() != "false"
                    players = prematch_service.list_players(league, active_only)
                    self._send_json(
                        {
                            "league": league.upper(),
                            "seasonCode": 2026,
                            "dataSource": prematch_service.source,
                            "players": players,
                        }
                    )
                except PrematchDataError as error:
                    self._send_json(
                        {"error": "prematch_data_unavailable", "detail": str(error)},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            if path.startswith("/api/v1/players/") and path.endswith("/image"):
                try:
                    player_code = unquote(path.removeprefix("/api/v1/players/").removesuffix("/image").strip("/"))
                    league = query.get("league", ["PBA"])[0]
                    image = prematch_service.repository.get_player_image(player_code, league)
                    if image is None:
                        self._send_json({"error": "image_not_found"}, HTTPStatus.NOT_FOUND)
                    else:
                        self._send_bytes(image[0], image[1])
                except PrematchDataError as error:
                    self._send_json(
                        {"error": "prematch_data_unavailable", "detail": str(error)},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            if path == "/api/v1/detection/latest":
                self._send_json(detections.get())
                return
            if path == "/api/v1/youtube/live/status":
                self._send_json(live_worker.status())
                return
            if path == "/api/v1/live-match-probability/latest":
                self._send_json(live_match.status())
                return
            if path in ("/assets/logo.png", "/assets/home.png"):
                image_path = ui_path.parent / "assets" / Path(path).name
                if image_path.exists():
                    self._send_bytes(image_path.read_bytes(), "image/png")
                else:
                    self._send_json({"error": "asset_not_found"}, HTTPStatus.NOT_FOUND)
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
            if path == "/extension-preview":
                body = (extension_dir / "sidepanel.html").read_bytes().replace(
                    b"<head>", b'<head><base href="/extension/">', 1
                )
                body = body.replace(
                    b"</head>",
                    b"<style>.app{max-width:480px;margin-left:auto;margin-right:auto}</style></head>",
                    1,
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/extension/"):
                filename = path.removeprefix("/extension/")
                allowed = {
                    "sidepanel.css": "text/css; charset=utf-8",
                    "sidepanel.js": "text/javascript; charset=utf-8",
                }
                if filename in allowed:
                    body = (extension_dir / filename).read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", allowed[filename])
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

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
                if path == "/api/v1/match-probability":
                    self._send_json(prematch_service.predict(payload))
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
                    detections.clear()
                    live_match.reset()
                    scoreboard_names.reset()
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
                if path == "/api/v1/youtube/live/sync":
                    live_worker.sync_to(float(payload["timestamp_seconds"]))
                    self._send_json(live_worker.status(), HTTPStatus.ACCEPTED)
                    return
                if path == "/api/v1/youtube/live/shooter":
                    live_worker.set_shooter(str(payload["shooter"]))
                    self._send_json(live_worker.status())
                    return
                if path == "/api/v1/live-match/players":
                    player_a = payload.get("player_a")
                    player_b = payload.get("player_b")
                    self._send_json(
                        live_match.set_player_names(
                            str(player_a) if player_a is not None else None,
                            str(player_b) if player_b is not None else None,
                        )
                    )
                    return
                if path == "/api/v1/youtube/live/scoreboard/reset":
                    scoreboard_names.reset()
                    live_worker.reset_scoreboard()
                    detections.clear_scoreboard()
                    self._send_json(live_worker.status())
                    return
                self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            except PrematchDataError as error:
                self._send_json(
                    {"error": "prematch_prediction_failed", "detail": str(error)},
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )
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
    parser.add_argument(
        "--shots",
        type=Path,
        default=root.parent / "billiard._public_._billiard_turns_.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "outputs" / "symmetric_hybrid_v2" / "model.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=root / "outputs" / "symmetric_hybrid_v2" / "calibration.json",
    )
    parser.add_argument("--ui", type=Path, default=root / "ui" / "index.html")
    parser.add_argument(
        "--extension",
        type=Path,
        default=root / "extension",
        help="Chrome extension directory and preview assets",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=root / "config" / "video1_table.json",
        help="Fixed overhead table calibration used by the broadcast detector",
    )
    parser.add_argument(
        "--ball-model",
        type=Path,
        default=root / "weights" / "best_3cls.pt",
        help="temp/minsu three-class YOLO ball detector",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    args = parse_args()
    service = ProbabilityService(
        args.shots.resolve(), args.model.resolve(), args.calibration.resolve()
    )
    detections = DetectionStore()
    video_analyzer = VideoPositionAnalyzer(
        table_config=args.table.resolve(), model_path=args.ball_model.resolve()
    )
    prematch_service = create_prematch_service()
    prematch_inputs = PrematchLiveInputProvider(prematch_service)
    scoreboard_names = LockedScoreboardPlayerMatcher(prematch_inputs)
    live_match = LiveMatchCoordinator(prematch_inputs)

    def publish_live_layout(
        positions: dict[str, tuple[float, float]],
        shooter: str,
        analysis: dict[str, object],
    ) -> None:
        before = {
            color: list(positions[f"{color}_ball"])
            for color in ("white", "yellow", "red")
        }
        confirmed = bool(analysis.get("confirmed"))
        shooter_confirmed = bool(analysis.get("shooterConfirmed"))
        # 공이 멈춰 레이아웃이 확정되면 수구가 아직 점수판으로 확정되지
        # 않았어도 곧바로 확률을 계산한다(잠정값). 이후 점수판이 원형 안
        # 숫자 색으로 수구를 확정하면 라이브 워커가 이 레이아웃으로 즉시
        # 재예측을 요청하므로, 여기서는 shooterConfirmed 플래그만 실어 준다.
        prediction = None
        if confirmed:
            prediction = service.predict(
                {
                    "before": before,
                    "shooter": shooter,
                    "position_error_mm": 25.0,
                }
            )
            prediction["shooterConfirmed"] = shooter_confirmed
            if shooter_confirmed:
                live_match.update_shot(
                    float(prediction["successProbability"]), shooter
                )
        detections.put(
            {
                "before": before,
                "shooter": shooter,
                "prediction": prediction,
                "confirmed": confirmed,
                "shooterConfirmed": shooter_confirmed,
                "analysis": analysis,
            }
        )

    def publish_scoreboard(scoreboard: dict[str, object]) -> None:
        try:
            resolved = scoreboard_names.match(scoreboard)
        except PrematchDataError:
            resolved = scoreboard
        detections.put_scoreboard(resolved)
        live_match.update_scoreboard(resolved)

    live_worker = YoutubeLiveWorker(
        video_analyzer,
        publish_live_layout,
        scoreboard_callback=publish_scoreboard,
    )
    handler = create_handler(
        service,
        prematch_service,
        args.ui.resolve(),
        args.extension.resolve(),
        detections,
        video_analyzer,
        live_worker,
        live_match,
        scoreboard_names,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"CueCast local UI: http://{args.host}:{args.port} | "
        f"records={len(service.records)} | model={service.model.version} | "
        f"prematch={prematch_service.source}"
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
