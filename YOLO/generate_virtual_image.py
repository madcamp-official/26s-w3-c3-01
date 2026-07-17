from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from render_board import draw_virtual_table


def extract_coordinates(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept result.json, state.json, or a direct ball-coordinate object."""
    if "balls" in payload and isinstance(payload["balls"], dict):
        return payload["balls"]
    if "coordinates" in payload and isinstance(payload["coordinates"], dict):
        return payload["coordinates"]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a virtual billiard-table image from normalized ball coordinates."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file containing red/yellow/white normalized coordinates.",
    )
    parser.add_argument(
        "--output",
        default="outputs/virtual_table.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The root JSON value must be an object.")

    coordinates = extract_coordinates(payload)
    draw_virtual_table(coordinates, args.output)
    print(f"Generated: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
