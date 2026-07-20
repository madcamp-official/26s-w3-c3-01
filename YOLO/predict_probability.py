from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuecast_yolo.shot_probability import (
    HybridShotProbabilityEngine,
    LayoutMm,
    layout_from_normalized_colors,
    load_continuous_model,
    load_shot_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a 3-cushion shot probability")
    parser.add_argument("state", help="JSON containing cueBall, before and optional quality")
    parser.add_argument("--shots", required=True, help="Historical shot records JSON")
    parser.add_argument("--model", required=True, help="Continuous model JSON")
    parser.add_argument("--out", default="outputs/probability/prediction.json")
    args = parser.parse_args()

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    records = load_shot_records(args.shots)
    model = load_continuous_model(args.model)
    engine = HybridShotProbabilityEngine(records, model)
    quality = state.get("quality") or {}
    if "shooter" in state:
        cue_ball = str(state["shooter"])
        layout, roles = layout_from_normalized_colors(state["before"], cue_ball)
        position_error_mm = float(state.get("position_error_mm", 25.0))
        prediction_id = (
            f"{state.get('video_id', 'state')}:turn:{state.get('turn', 0)}:"
            f"epoch:{state.get('epoch', 0)}"
        )
    else:
        cue_ball = str(state["cueBall"])
        layout = LayoutMm.from_dict(state["before"])
        object_colors = sorted(color for color in ("white", "yellow", "red") if color != cue_ball)
        roles = (cue_ball, object_colors[0], object_colors[1])
        position_error_mm = float(quality.get("positionErrorMm", 25.0))
        prediction_id = state.get("predictionId")
    result = engine.predict(
        layout,
        cue_ball,
        position_error_mm=position_error_mm,
        prediction_id=prediction_id,
    )
    result["roles"] = {
        "cue": roles[0],
        "object1": roles[1],
        "object2": roles[2],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
