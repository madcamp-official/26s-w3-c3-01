from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cuecast_yolo.shot_probability import load_shot_records
from cuecast_yolo.symmetric_probability import (
    DEFAULT_SYMMETRIC_GRID,
    PlattCalibration,
    SymmetricCatBoostCoordinateModel,
    SymmetricHybridShotProbabilityEngine,
    fit_platt_calibration,
    save_calibration,
)


def group_id(shot_id: str) -> str:
    return shot_id.split(":turn:", 1)[0]


def probability_metrics(
    outcomes: list[float], probabilities: list[float]
) -> dict[str, float]:
    y = np.asarray(outcomes, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "logLoss": float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))),
        "brierScore": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p >= 0.5) == y)),
    }


def fit_grouped_calibration(records: list[object]) -> tuple[PlattCalibration, dict[str, object]]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for record in records:
        grouped[group_id(record.shot_id)].append(record)  # type: ignore[attr-defined]
    outcomes: list[float] = []
    raw_probabilities: list[float] = []
    fold_rows: list[dict[str, object]] = []
    for held_out, test_records in grouped.items():
        training_records = [
            record
            for name, group_records in grouped.items()
            if name != held_out
            for record in group_records
        ]
        model = SymmetricCatBoostCoordinateModel.fit(training_records)
        engine = SymmetricHybridShotProbabilityEngine(
            training_records,
            model,
            grid_config=DEFAULT_SYMMETRIC_GRID,
            calibration=PlattCalibration(),
        )
        fold_outcomes: list[float] = []
        fold_probabilities: list[float] = []
        for record in test_records:
            result = engine._predict_once(record.before, record.cue_ball)  # type: ignore[attr-defined]
            outcome = float(record.success)  # type: ignore[attr-defined]
            probability = float(result["rawProbability"])
            outcomes.append(outcome)
            raw_probabilities.append(probability)
            fold_outcomes.append(outcome)
            fold_probabilities.append(probability)
        fold_rows.append(
            {
                "group": held_out,
                "records": len(test_records),
                **probability_metrics(fold_outcomes, fold_probabilities),
            }
        )
    calibration = fit_platt_calibration(raw_probabilities, outcomes)
    calibrated = [calibration.apply(value) for value in raw_probabilities]
    report = {
        "method": "leave-one-video-out",
        "groups": len(grouped),
        "raw": probability_metrics(outcomes, raw_probabilities),
        "calibrated": probability_metrics(outcomes, calibrated),
        "calibration": calibration.to_dict(),
        "folds": fold_rows,
    }
    return calibration, report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train Symmetric Hybrid v2")
    parser.add_argument(
        "shots", nargs="?", type=Path, default=root.parent / "billiard_turns_export.jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=root / "outputs" / "symmetric_hybrid_v2"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_shot_records(args.shots.resolve())
    if len({record.success for record in records}) < 2:
        raise ValueError("Symmetric CatBoost requires success and failure records")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    calibration, validation = fit_grouped_calibration(records)
    model = SymmetricCatBoostCoordinateModel.fit(records)
    model.save(output / "model.json")
    save_calibration(calibration, output / "calibration.json")

    engine = SymmetricHybridShotProbabilityEngine(
        records,
        model,
        grid_config=DEFAULT_SYMMETRIC_GRID,
        calibration=calibration,
    )
    selected_levels = Counter(
        engine.grid.selected_level(record.before) for record in records
    )
    split_parents: list[int] = []
    for child_level in (1, 2):
        split_parents.append(
            sum(
                bool(engine.grid.split_diagnostics(key, child_level)["splitWorthy"])
                for key, stat in engine.grid.levels[child_level - 1].items()
                if stat.attempts
                >= DEFAULT_SYMMETRIC_GRID.minimum_samples_for_split
            )
        )
    report = {
        "engineVersion": "symmetric-hybrid-v2",
        "records": len(records),
        "successes": sum(record.success for record in records),
        "modelVersion": model.version,
        "model": {
            "iterations": 75,
            "depth": 3,
            "learningRate": 0.03,
            "l2LeafReg": 30.0,
            "randomStrength": 1.0,
            "autoClassWeights": None,
        },
        "symmetryCount": 8,
        "grid": {
            "dimensions": [[4, 2], [8, 4], [16, 8]],
            "minimumParentSamples": DEFAULT_SYMMETRIC_GRID.minimum_samples_for_split,
            "minimumChildSamples": DEFAULT_SYMMETRIC_GRID.minimum_child_samples,
            "selectedLevelCounts": {
                str(level): selected_levels.get(level, 0) for level in range(3)
            },
            "splitParentCounts": {"level1": split_parents[0], "level2": split_parents[1]},
        },
        "validation": validation,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
