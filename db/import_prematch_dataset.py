"""Load final_dataset_2026_start.zip into the CueCast PostgreSQL database.

Usage:
  python db/import_prematch_dataset.py final_dataset_2026_start.zip

DATABASE_URL is read from the environment unless --database-url is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y", "t"}


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(root: Path, relative: str) -> list[dict[str, str]]:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_dataset_root(path: Path) -> Path:
    marker = Path("02_runtime/player_runtime_state_2026_start.csv")
    if (path / marker).exists():
        return path
    matches = [candidate for candidate in path.glob("final_dataset*") if (candidate / marker).exists()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Could not identify one dataset root below {path}")
    return matches[0]


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (destination / entry.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
        bundle.extractall(destination)


@contextmanager
def dataset_directory(source: Path) -> Iterator[Path]:
    if source.is_dir():
        yield find_dataset_root(source)
        return
    if source.suffix.casefold() != ".zip":
        raise ValueError("Dataset must be an extracted directory or .zip archive")
    with tempfile.TemporaryDirectory(prefix="cuecast-prematch-") as temp_dir:
        temp = Path(temp_dir)
        safe_extract(source, temp)
        yield find_dataset_root(temp)


def load_dataset(connection, root: Path, *, include_images: bool) -> dict[str, int]:
    from psycopg2.extras import Json, execute_values

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8-sig"))
    snapshot_at = metadata["build_time_utc"]
    players = read_csv(root, "01_players/player_master_2026.csv")
    runtime = read_csv(root, "02_runtime/player_runtime_state_2026_start.csv")
    season_rows = read_csv(root, "03_current_season/season_player_state_2026.csv")
    h2h = read_csv(root, "02_runtime/head_to_head_reference_2026_start.csv")
    baselines = read_csv(root, "02_runtime/league_metric_baselines_2026_start.csv")
    season_index = {
        (row["league"], as_int(row["season_code"]), row["player_code"]): row
        for row in season_rows
    }

    player_values = []
    for row in players:
        image_path = root / row["image_file"]
        image_bytes = image_path.read_bytes() if include_images and image_path.exists() else None
        image_mime = mimetypes.guess_type(image_path.name)[0] if image_bytes else None
        player_values.append(
            (
                row["league"],
                row["player_code"],
                row["player_name"],
                row["player_name_short"],
                as_bool(row["active_2026_roster"]),
                as_bool(row["image_is_placeholder"]),
                row["image_file"],
                image_mime,
                image_bytes,
            )
        )

    feature_values = []
    for row in runtime:
        key = (row["league"], as_int(row["season_code"]), row["player_code"])
        season = season_index.get(key, {})
        season_matches = as_int(season.get("matches", row.get("season_matches")))
        season_wins = as_int(season.get("wins", row.get("season_wins")))
        metrics = {
            "AVG": as_optional_float(row.get("final_avg_start")),
            "TS": as_optional_float(row.get("final_ts_pct_start")),
            "BRS": as_optional_float(row.get("final_brs_pct_start")),
            "5HS": as_optional_float(row.get("final_5hs_pct_start")),
            "HR": as_optional_float(row.get("final_high_run_start")),
        }
        feature_values.append(
            (
                snapshot_at,
                row["league"],
                as_int(row["season_code"]),
                row["player_code"],
                as_float(season.get("elo_current", row.get("elo_start")), 1500.0),
                as_int(row.get("career_matches_prior")) + season_matches,
                as_int(row.get("career_wins_prior")) + season_wins,
                season_matches,
                season_wins,
                as_int(row.get("last5_matches_prior")),
                as_int(row.get("last5_wins_prior")),
                as_int(row.get("last10_matches_prior")),
                as_int(row.get("last10_wins_prior")),
                as_float(row.get("performance_q_start")),
                as_float(row.get("prior_innings")) + as_float(season.get("innings")),
                Json(metrics),
                row.get("snapshot_cutoff_rule"),
            )
        )

    h2h_values = [
        (
            row["league"],
            row["player_a_code"],
            row["player_b_code"],
            as_int(row["played_matches"]),
            as_int(row["player_a_wins"]),
            as_int(row["player_b_wins"]),
            row.get("first_match_datetime") or None,
            row.get("last_match_datetime") or None,
            row.get("latest_winner_code") or None,
            row.get("latest_winner_name") or None,
            row.get("latest_match_uid") or None,
        )
        for row in h2h
    ]
    baseline_values = [
        (
            snapshot_at,
            row["league"],
            as_int(row["season_code"]),
            row["metric"],
            as_optional_float(row.get("prior_league_mean")),
            as_optional_float(row.get("standardization_mean")),
            as_optional_float(row.get("standardization_std_population")),
            as_int(row.get("standardization_roster_count")),
        )
        for row in baselines
    ]

    with connection.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO prematch_players
              (league, player_code, player_name, player_name_short, active_roster,
               image_is_placeholder, image_source_path, image_mime_type, image_bytes)
            VALUES %s
            ON CONFLICT (league, player_code) DO UPDATE SET
              player_name = EXCLUDED.player_name,
              player_name_short = EXCLUDED.player_name_short,
              active_roster = EXCLUDED.active_roster,
              image_is_placeholder = EXCLUDED.image_is_placeholder,
              image_source_path = EXCLUDED.image_source_path,
              image_mime_type = COALESCE(EXCLUDED.image_mime_type, prematch_players.image_mime_type),
              image_bytes = COALESCE(EXCLUDED.image_bytes, prematch_players.image_bytes),
              updated_at = now()
            """,
            player_values,
            page_size=200,
        )
        execute_values(
            cursor,
            """
            INSERT INTO prematch_player_features
              (snapshot_at, league, season_code, player_code, elo, career_matches,
               career_wins, season_matches, season_wins, last5_matches, last5_wins,
               last10_matches, last10_wins, performance_score,
               performance_innings_total, metrics, source_cutoff_rule)
            VALUES %s
            ON CONFLICT (snapshot_at, league, player_code) DO UPDATE SET
              elo = EXCLUDED.elo,
              career_matches = EXCLUDED.career_matches,
              career_wins = EXCLUDED.career_wins,
              season_matches = EXCLUDED.season_matches,
              season_wins = EXCLUDED.season_wins,
              last5_matches = EXCLUDED.last5_matches,
              last5_wins = EXCLUDED.last5_wins,
              last10_matches = EXCLUDED.last10_matches,
              last10_wins = EXCLUDED.last10_wins,
              performance_score = EXCLUDED.performance_score,
              performance_innings_total = EXCLUDED.performance_innings_total,
              metrics = EXCLUDED.metrics,
              source_cutoff_rule = EXCLUDED.source_cutoff_rule
            """,
            feature_values,
            page_size=200,
        )
        execute_values(
            cursor,
            """
            INSERT INTO prematch_head_to_head
              (league, player_a_code, player_b_code, played_matches, player_a_wins,
               player_b_wins, first_match_datetime, last_match_datetime,
               latest_winner_code, latest_winner_name, latest_match_uid)
            VALUES %s
            ON CONFLICT (league, player_a_code, player_b_code) DO UPDATE SET
              played_matches = EXCLUDED.played_matches,
              player_a_wins = EXCLUDED.player_a_wins,
              player_b_wins = EXCLUDED.player_b_wins,
              first_match_datetime = EXCLUDED.first_match_datetime,
              last_match_datetime = EXCLUDED.last_match_datetime,
              latest_winner_code = EXCLUDED.latest_winner_code,
              latest_winner_name = EXCLUDED.latest_winner_name,
              latest_match_uid = EXCLUDED.latest_match_uid
            """,
            h2h_values,
            page_size=500,
        )
        execute_values(
            cursor,
            """
            INSERT INTO prematch_league_metric_baselines
              (snapshot_at, league, season_code, metric, prior_league_mean,
               standardization_mean, standardization_std_population,
               standardization_roster_count)
            VALUES %s
            ON CONFLICT (snapshot_at, league, metric) DO UPDATE SET
              prior_league_mean = EXCLUDED.prior_league_mean,
              standardization_mean = EXCLUDED.standardization_mean,
              standardization_std_population = EXCLUDED.standardization_std_population,
              standardization_roster_count = EXCLUDED.standardization_roster_count
            """,
            baseline_values,
        )

    return {
        "players": len(player_values),
        "features": len(feature_values),
        "head_to_head": len(h2h_values),
        "baselines": len(baseline_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import CueCast pre-match dataset")
    parser.add_argument("dataset", type=Path, help="Dataset directory or ZIP archive")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not store player image bytes in PostgreSQL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    try:
        import psycopg2
    except ImportError as error:
        raise SystemExit("Install db/requirements.txt before importing") from error

    schema = Path(__file__).with_name("prematch_schema.sql").read_text(encoding="utf-8")
    with dataset_directory(args.dataset.resolve()) as root:
        with psycopg2.connect(args.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(schema)
            counts = load_dataset(connection, root, include_images=not args.skip_images)
        print(json.dumps({"dataset": str(root), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
