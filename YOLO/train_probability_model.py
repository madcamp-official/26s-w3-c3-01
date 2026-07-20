from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuecast_yolo.shot_probability import (
    BootstrapProbabilityModel,
    CatBoostCoordinateModel,
    LogisticCoordinateModel,
    load_shot_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the continuous shot model")
    parser.add_argument("shots", help="Shot records JSON")
    parser.add_argument("--out", default="outputs/probability/model.json")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument(
        "--model-type",
        choices=("auto", "bootstrap", "logistic", "catboost"),
        default="auto",
    )
    parser.add_argument("--bootstrap-probability", type=float, default=0.5)
    parser.add_argument("--bootstrap-strength", type=float, default=20.0)
    args = parser.parse_args()

    records = load_shot_records(args.shots)
    model_type = args.model_type
    outcomes = {record.success for record in records}
    if model_type == "auto":
        if not records:
            model_type = "bootstrap"
        elif (
            len(records) >= 50
            and len(outcomes) == 2
            and CatBoostCoordinateModel.is_available()
        ):
            model_type = "catboost"
        else:
            model_type = "logistic"

    if model_type == "bootstrap":
        if records:
            successes = sum(record.success for record in records)
            probability = (
                successes
                + args.bootstrap_strength * args.bootstrap_probability
            ) / (len(records) + args.bootstrap_strength)
        else:
            probability = args.bootstrap_probability
        model = BootstrapProbabilityModel(probability)
    elif model_type == "logistic":
        model = LogisticCoordinateModel.fit(records, epochs=args.epochs)
    else:
        model = CatBoostCoordinateModel.fit(records, iterations=args.iterations)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    print(
        json.dumps(
            {
                "records": len(records),
                "modelType": model_type,
                "modelVersion": model.version,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
