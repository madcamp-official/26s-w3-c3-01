from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from cuecast_yolo.shot_probability import (
    GridConfig,
    HybridConfig,
    HybridShotProbabilityEngine,
    LayoutMm,
    LogisticCoordinateModel,
    PointMm,
    ordered_object_colors,
    load_continuous_model,
    load_shot_records,
)


# This experiment intentionally lives outside the production engine defaults.
EXPERIMENTAL_GRID = GridConfig(
    base_columns=4,
    base_rows=2,
    maximum_level=2,
    minimum_samples_for_split=3,
    preferred_samples_for_split=6,
    minimum_child_samples=1,
    minimum_probability_difference=0.25,
    minimum_validation_logloss_improvement=0.02,
    prior_strengths=(16.0, 10.0, 7.0),
)

# Platt calibration fitted from leave-one-video-out predictions on the 503-shot export.
CALIBRATION_INTERCEPT = 0.0318597921158656
CALIBRATION_SLOPE = 0.37335537577217154


def calibrate_probability(probability: float) -> float:
    clipped = min(1.0 - 1e-7, max(1e-7, probability))
    logit = math.log(clipped / (1.0 - clipped))
    value = CALIBRATION_INTERCEPT + CALIBRATION_SLOPE * logit
    return 1.0 / (1.0 + math.exp(-value))


def calibrated_standard_deviation(probability: float, standard_deviation: float) -> float:
    calibrated = calibrate_probability(probability)
    clipped = min(1.0 - 1e-7, max(1e-7, probability))
    derivative = (
        CALIBRATION_SLOPE
        * calibrated
        * (1.0 - calibrated)
        / (clipped * (1.0 - clipped))
    )
    return min(0.5, max(0.0, abs(derivative) * standard_deviation))


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


class ExperimentalAdaptiveService:
    def __init__(self, shots_path: Path, model_path: Path) -> None:
        self.records = load_shot_records(shots_path)
        if model_path.exists():
            self.model = load_continuous_model(model_path)
        else:
            self.model = LogisticCoordinateModel.fit(self.records)
        self.config = HybridConfig(grid=EXPERIMENTAL_GRID)
        self.engine = HybridShotProbabilityEngine(self.records, self.model, self.config)
        self.summary = self._grid_summary()

    def _grid_summary(self) -> dict[str, object]:
        selected = Counter(self.engine.grid.selected_level(record.before) for record in self.records)
        split_parents: list[list[dict[str, object]]] = [[], []]
        for child_level in (1, 2):
            for key, stat in self.engine.grid.levels[child_level - 1].items():
                if stat.attempts < EXPERIMENTAL_GRID.minimum_samples_for_split:
                    continue
                diagnostic = self.engine.grid.split_diagnostics(key, child_level)
                if bool(diagnostic["splitWorthy"]):
                    split_parents[child_level - 1].append(
                        {"parentKey": list(key), **diagnostic}
                    )
        return {
            "baseGrid": [EXPERIMENTAL_GRID.base_columns, EXPERIMENTAL_GRID.base_rows],
            "levelDimensions": [
                list(self.engine.grid.dimensions(level)) for level in range(3)
            ],
            "occupiedStates": [len(level) for level in self.engine.grid.levels],
            "selectedLevelCounts": {
                str(level): selected.get(level, 0) for level in range(3)
            },
            "splitWorthyParents": split_parents,
            "thresholds": {
                "minimumParentSamples": EXPERIMENTAL_GRID.minimum_samples_for_split,
                "preferredParentSamples": EXPERIMENTAL_GRID.preferred_samples_for_split,
                "minimumChildSamples": EXPERIMENTAL_GRID.minimum_child_samples,
                "minimumProbabilityDifference": EXPERIMENTAL_GRID.minimum_probability_difference,
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
                "shooter": record.cue_ball,
                "success": record.success,
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
            selected_level = self.engine.grid.selected_level(record.before)
            if selected_level < 1:
                break
            parent_key = self.engine.grid.state_key(record.before, selected_level - 1)
            identity = (selected_level, parent_key)
            if identity in seen:
                continue
            seen.add(identity)
            covered = sum(
                self.engine.grid.selected_level(candidate.before) == selected_level
                and self.engine.grid.state_key(candidate.before, selected_level - 1)
                == parent_key
                for candidate in self.records
            )
            examples.append(
                {
                    "selectedLevel": selected_level,
                    "parentKey": list(parent_key),
                    "coveredRecords": covered,
                    "shooter": record.cue_ball,
                    "formation": self._normalized_roles(record),
                }
            )
        return {"shots": shots, "grid": self.summary, "splitExamples": examples}

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "records": len(self.records),
            "modelVersion": self.model.version,
            "engine": "experimental-calibrated-logistic-hybrid-v1",
            "grid": self.summary,
            "validation": {
                "method": "leave-one-video-out",
                "logLoss": 0.680760,
                "brierScore": 0.243740,
            },
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, object]:
        shooter = str(payload.get("shooter", "yellow"))
        if shooter not in ("white", "yellow"):
            raise ValueError("shooter는 white 또는 yellow여야 합니다")
        formation = payload.get("formation")
        if not isinstance(formation, dict):
            raise ValueError("formation 객체가 필요합니다")
        layout = layout_from_roles(formation)
        result = self.engine.predict(
            layout,
            shooter,
            position_error_mm=float(payload.get("position_error_mm", 25.0)),
            prediction_id=payload.get("prediction_id", "experimental-web"),
        )
        raw_probability = float(result["successProbability"])
        raw_standard_deviation = float(result["uncertainty"]["standardDeviation"])  # type: ignore[index]
        calibrated = calibrate_probability(raw_probability)
        result["rawSuccessProbability"] = raw_probability
        result["successProbability"] = calibrated
        result["difficulty"] = 1.0 - calibrated
        result["uncertainty"]["rawStandardDeviation"] = raw_standard_deviation  # type: ignore[index]
        result["uncertainty"]["standardDeviation"] = calibrated_standard_deviation(  # type: ignore[index]
            raw_probability, raw_standard_deviation
        )
        object1, object2 = ordered_object_colors(shooter)
        result["roles"] = {"cue": shooter, "object1": object1, "object2": object2}
        result["experimental"] = {
            "engine": "experimental-calibrated-logistic-hybrid-v1",
            "calibration": {
                "intercept": CALIBRATION_INTERCEPT,
                "slope": CALIBRATION_SLOPE,
            },
            "grid": self.summary,
        }
        result["dataRecords"] = len(self.records)
        return result


def create_handler(
    service: ExperimentalAdaptiveService, ui_path: Path
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

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/v1/health":
                self._send_json(service.health())
                return
            if path == "/api/v1/data":
                self._send_json(service.data())
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
            try:
                if self.path.split("?", 1)[0] != "/api/v1/shot-probability":
                    self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
                    return
                self._send_json(service.predict(self._read_json()))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(
                    {"error": "invalid_request", "detail": str(error)},
                    HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: object) -> None:
            print(f"[experimental-adaptive] {self.address_string()} {format % args}")

    return Handler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Experimental adaptive-grid web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--shots", type=Path, default=root.parent / "billiard_turns_export.jsonl")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "outputs" / "hybrid_billiard_turns" / "logistic_model.json",
    )
    parser.add_argument(
        "--ui", type=Path, default=root / "ui" / "experimental_adaptive.html"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = ExperimentalAdaptiveService(args.shots.resolve(), args.model.resolve())
    server = ThreadingHTTPServer(
        (args.host, args.port), create_handler(service, args.ui.resolve())
    )
    print(
        f"Experimental Adaptive Grid: http://{args.host}:{args.port} | "
        f"records={len(service.records)} | model={service.model.version}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
