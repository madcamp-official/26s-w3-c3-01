from __future__ import annotations

import argparse
import json
from pathlib import Path

from cuecast_yolo.shot_probability import TABLE_HEIGHT_MM, TABLE_WIDTH_MM


def point(row: dict[str, object], phase: str, index: int) -> dict[str, float]:
    return {
        "xMm": round(float(row[f"{phase}_ball_{index}_x"]) * TABLE_WIDTH_MM, 3),
        "yMm": round(float(row[f"{phase}_ball_{index}_y"]) * TABLE_HEIGHT_MM, 3),
    }


def convert_row(row: dict[str, object], prefix: str) -> dict[str, object]:
    cue_name = str(row["cue_ball"])
    cue_color = cue_name.removesuffix("_ball")
    return {
        "shotId": f"{prefix}_shot_{row['shot_id']}",
        "cueBall": cue_color,
        "before": {
            "cue": point(row, "start", 1),
            "object1": point(row, "start", 2),
            "object2": point(row, "start", 3),
        },
        "after": {
            "cue": point(row, "end", 1),
            "object1": point(row, "end", 2),
            "object2": point(row, "end", 3),
        },
        "success": bool(row["success"]),
        "points": int(bool(row["success"])),
        "player": {"playerId": None, "avg": None},
        "quality": {
            "positionErrorMm": 25.0,
            "detectionConfidence": round(float(row.get("cue_confidence", 1.0)), 4),
            "sourceQuality": row.get("quality"),
            "reviewNote": row.get("review_note"),
        },
        "source": {
            "startTimestampSeconds": row.get("start_timestamp"),
            "movementTimestampSeconds": row.get("movement_timestamp"),
            "endTimestampSeconds": row.get("end_timestamp"),
            "scoreChangeDistance": row.get("score_change_distance"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert extracted normalized shots to probability-model records"
    )
    parser.add_argument("shots", help="extract_shot_dataset.py shots.json")
    parser.add_argument("--prefix", required=True, help="Unique match id")
    parser.add_argument("--out", default="outputs/probability/shots.json")
    parser.add_argument(
        "--include-review", action="store_true", help="Include records marked for review"
    )
    args = parser.parse_args()

    rows = json.loads(Path(args.shots).read_text(encoding="utf-8"))
    accepted = [
        row
        for row in rows
        if row.get("cue_ball")
        and isinstance(row.get("success"), bool)
        and (args.include_review or row.get("quality") == "ready")
    ]
    converted = [convert_row(row, args.prefix) for row in accepted]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "inputRecords": len(rows),
                "acceptedRecords": len(converted),
                "rejectedRecords": len(rows) - len(converted),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
