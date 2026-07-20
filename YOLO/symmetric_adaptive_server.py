from __future__ import annotations

import argparse
import json
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cuecast_yolo.shot_probability import LayoutMm, PointMm, load_shot_records, ordered_object_colors
from cuecast_yolo.symmetric_probability import (
    DEFAULT_SYMMETRIC_GRID,
    SymmetricCatBoostCoordinateModel,
    SymmetricHybridShotProbabilityEngine,
    load_calibration,
)


def layout_from_roles(value: dict[str, object]) -> LayoutMm:
    points: list[PointMm] = []
    for role in ("cue", "object1", "object2"):
        raw = value.get(role)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"{role} 좌표는 [x, y] 형식이어야 합니다")
        x, y = float(raw[0]), float(raw[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{role} 좌표는 0~1 범위여야 합니다")
        points.append(PointMm(x * 2840.0, y * 1420.0))
    return LayoutMm(*points)


class SymmetricProbabilityService:
    def __init__(
        self,
        shots_path: Path,
        model_path: Path,
        calibration_path: Path,
        report_path: Path,
    ) -> None:
        self.records = load_shot_records(shots_path)
        self.model = SymmetricCatBoostCoordinateModel.load(model_path)
        self.calibration = load_calibration(calibration_path)
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.engine = SymmetricHybridShotProbabilityEngine(
            self.records,
            self.model,
            grid_config=DEFAULT_SYMMETRIC_GRID,
            calibration=self.calibration,
        )
        self.summary = self._grid_summary()

    def _grid_summary(self) -> dict[str, object]:
        selected = Counter(self.engine.grid.selected_level(record.before) for record in self.records)
        split_parents: list[list[dict[str, object]]] = [[], []]
        for child_level in (1, 2):
            for key, stat in self.engine.grid.levels[child_level - 1].items():
                if stat.attempts < DEFAULT_SYMMETRIC_GRID.minimum_samples_for_split:
                    continue
                diagnostic = self.engine.grid.split_diagnostics(key, child_level)
                if bool(diagnostic["splitWorthy"]):
                    split_parents[child_level - 1].append(
                        {"parentKey": list(key), **diagnostic}
                    )
        return {
            "dimensions": [[4, 2], [8, 4], [16, 8]],
            "occupiedStates": [len(level) for level in self.engine.grid.levels],
            "selectedLevelCounts": {str(level): selected.get(level, 0) for level in range(3)},
            "splitWorthyParents": split_parents,
            "thresholds": {
                "minimumParentSamples": DEFAULT_SYMMETRIC_GRID.minimum_samples_for_split,
                "preferredParentSamples": DEFAULT_SYMMETRIC_GRID.preferred_samples_for_split,
                "minimumChildSamples": DEFAULT_SYMMETRIC_GRID.minimum_child_samples,
                "minimumProbabilityDifference": DEFAULT_SYMMETRIC_GRID.minimum_probability_difference,
            },
        }

    @staticmethod
    def _normalized_roles(record: object) -> dict[str, list[float]]:
        layout = record.before  # type: ignore[attr-defined]
        return {
            "cue": [layout.cue.x_mm / 2840.0, layout.cue.y_mm / 1420.0],
            "object1": [layout.object1.x_mm / 2840.0, layout.object1.y_mm / 1420.0],
            "object2": [layout.object2.x_mm / 2840.0, layout.object2.y_mm / 1420.0],
        }

    def data(self) -> dict[str, object]:
        shots = [
            {
                "id": record.shot_id,
                "success": record.success,
                "shooter": record.cue_ball,
                **self._normalized_roles(record),
            }
            for record in self.records
        ]
        examples: list[dict[str, object]] = []
        seen: set[tuple[int, tuple[int, int, int]]] = set()
        ranked = sorted(
            self.records,
            key=lambda record: self.engine.grid.selected_level(record.before),
            reverse=True,
        )
        for record in ranked:
            level = self.engine.grid.selected_level(record.before)
            if level < 1:
                break
            parent_key = self.engine.grid.state_key(record.before, level - 1)
            identity = (level, parent_key)
            if identity in seen:
                continue
            seen.add(identity)
            examples.append(
                {
                    "selectedLevel": level,
                    "parentKey": list(parent_key),
                    "shooter": record.cue_ball,
                    "formation": self._normalized_roles(record),
                }
            )
        return {"shots": shots, "grid": self.summary, "splitExamples": examples}

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "records": len(self.records),
            "engineVersion": "symmetric-hybrid-v2",
            "modelVersion": self.model.version,
            "modelConfig": self.report["model"],
            "symmetryCount": 8,
            "calibration": self.calibration.to_dict(),
            "grid": self.summary,
            "validation": self.report["validation"]["calibrated"],
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, object]:
        shooter = str(payload.get("shooter", "yellow"))
        if shooter not in ("white", "yellow"):
            raise ValueError("shooter는 white 또는 yellow여야 합니다")
        formation = payload.get("formation")
        if not isinstance(formation, dict):
            raise ValueError("formation 객체가 필요합니다")
        result = self.engine.predict(
            layout_from_roles(formation),
            shooter,
            position_error_mm=float(payload.get("position_error_mm", 25.0)),
            prediction_id=payload.get("prediction_id", "symmetric-web"),
        )
        object1, object2 = ordered_object_colors(shooter)
        result["roles"] = {"cue": shooter, "object1": object1, "object2": object2}
        result["dataRecords"] = len(self.records)
        return result


def create_handler(service: SymmetricProbabilityService, ui_path: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _payload(self) -> dict[str, Any]:
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
                self._json(service.health())
            elif path == "/api/v1/data":
                self._json(service.data())
            elif path in ("/", "/index.html"):
                body = ui_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path.split("?", 1)[0] != "/api/v1/shot-probability":
                    self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json(service.predict(self._payload()))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json({"error": "invalid_request", "detail": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[symmetric-v2] {self.address_string()} {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    output = root / "outputs" / "symmetric_hybrid_v2"
    parser = argparse.ArgumentParser(description="Serve Symmetric Hybrid v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--shots", type=Path, default=root.parent / "billiard_turns_export.jsonl")
    parser.add_argument("--model", type=Path, default=output / "model.json")
    parser.add_argument("--calibration", type=Path, default=output / "calibration.json")
    parser.add_argument("--report", type=Path, default=output / "training_report.json")
    parser.add_argument("--ui", type=Path, default=root / "ui" / "symmetric_adaptive.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = SymmetricProbabilityService(
        args.shots.resolve(),
        args.model.resolve(),
        args.calibration.resolve(),
        args.report.resolve(),
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(service, args.ui.resolve()))
    print(f"Symmetric Hybrid v2: http://{args.host}:{args.port} | records={len(service.records)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
