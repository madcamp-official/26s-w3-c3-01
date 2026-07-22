from __future__ import annotations

import csv
import json
import math
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


class PrematchDataError(ValueError):
    pass


@dataclass(frozen=True)
class PrematchConfig:
    elo_scale: float = 400.0
    elo_confidence_matches: int = 10
    career_prior_matches: int = 20
    season_prior_matches: int = 10
    recent5_prior_matches: int = 5
    recent10_prior_matches: int = 10
    performance_full_confidence_innings: int = 150
    performance_logistic_scale: float = 0.75
    category_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "elo": 0.30,
            "career": 0.175,
            "season": 0.05,
            "recent": 0.025,
            "performance": 0.45,
        }
    )


@dataclass(frozen=True)
class PlayerFeatureSnapshot:
    league: str
    season_code: int
    player_code: str
    player_name: str
    player_name_short: str
    active_roster: bool
    image_is_placeholder: bool
    elo: float
    career_matches: int
    career_wins: int
    season_matches: int
    season_wins: int
    last5_matches: int
    last5_wins: int
    last10_matches: int
    last10_wins: int
    performance_score: float
    performance_innings_total: float
    metrics: Mapping[str, float | None] = field(default_factory=dict)


class PrematchRepository(Protocol):
    source: str

    def list_players(self, league: str, active_only: bool = True) -> list[dict[str, Any]]: ...

    def get_snapshot(
        self, player_code: str, league: str, season_code: int
    ) -> PlayerFeatureSnapshot: ...

    def get_head_to_head(
        self, player_a_code: str, player_b_code: str, league: str
    ) -> dict[str, Any] | None: ...

    def get_player_image(self, player_code: str, league: str) -> tuple[bytes, str] | None: ...


def _as_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y", "t"}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clip_probability(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _safe_logistic(value: float) -> float:
    bounded = min(max(value, -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-bounded))


def adjusted_rate(wins: int, matches: int, threshold: int) -> float:
    if matches <= 0:
        return 0.5
    if matches < threshold:
        return (wins + 0.5 * (threshold - matches)) / threshold
    return wins / matches


def pair_probability(rate_a: float, rate_b: float) -> float:
    numerator = rate_a * (1.0 - rate_b)
    denominator = numerator + rate_b * (1.0 - rate_a)
    if abs(denominator) < 1e-15:
        return 0.5
    return _clip_probability(numerator / denominator)


def predict_from_snapshots(
    player_a: PlayerFeatureSnapshot,
    player_b: PlayerFeatureSnapshot,
    *,
    head_to_head: Mapping[str, Any] | None = None,
    config: PrematchConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PrematchConfig()
    if player_a.player_code == player_b.player_code:
        raise PrematchDataError("서로 다른 두 선수를 선택해 주세요")
    if player_a.league != player_b.league:
        raise PrematchDataError("PBA와 LPBA 선수는 서로 분리해 계산해야 합니다")
    if player_a.season_code != player_b.season_code:
        raise PrematchDataError("같은 시즌의 선수 스냅샷이 필요합니다")

    p_elo = 1.0 / (1.0 + 10.0 ** ((player_b.elo - player_a.elo) / cfg.elo_scale))
    c_elo = 0.5 + 0.5 * min(
        min(player_a.career_matches, player_b.career_matches)
        / cfg.elo_confidence_matches,
        1.0,
    )

    career_a = adjusted_rate(
        player_a.career_wins, player_a.career_matches, cfg.career_prior_matches
    )
    career_b = adjusted_rate(
        player_b.career_wins, player_b.career_matches, cfg.career_prior_matches
    )
    p_career = pair_probability(career_a, career_b)
    c_career = min(
        min(player_a.career_matches, player_b.career_matches)
        / cfg.career_prior_matches,
        1.0,
    )

    season_a = adjusted_rate(
        player_a.season_wins, player_a.season_matches, cfg.season_prior_matches
    )
    season_b = adjusted_rate(
        player_b.season_wins, player_b.season_matches, cfg.season_prior_matches
    )
    p_season = pair_probability(season_a, season_b)
    c_season = min(
        min(player_a.season_matches, player_b.season_matches)
        / cfg.season_prior_matches,
        1.0,
    )

    last5_a = adjusted_rate(
        player_a.last5_wins, player_a.last5_matches, cfg.recent5_prior_matches
    )
    last5_b = adjusted_rate(
        player_b.last5_wins, player_b.last5_matches, cfg.recent5_prior_matches
    )
    last10_a = adjusted_rate(
        player_a.last10_wins, player_a.last10_matches, cfg.recent10_prior_matches
    )
    last10_b = adjusted_rate(
        player_b.last10_wins, player_b.last10_matches, cfg.recent10_prior_matches
    )
    p_last5 = pair_probability(last5_a, last5_b)
    p_last10 = pair_probability(last10_a, last10_b)
    c_last5 = min(
        min(player_a.last5_matches, player_b.last5_matches)
        / cfg.recent5_prior_matches,
        1.0,
    )
    c_last10 = min(
        min(player_a.last10_matches, player_b.last10_matches)
        / cfg.recent10_prior_matches,
        1.0,
    )
    c_recent = 0.4 * c_last5 + 0.6 * c_last10
    p_recent = (
        (0.4 * c_last5 * p_last5 + 0.6 * c_last10 * p_last10) / c_recent
        if c_recent > 0
        else 0.5
    )

    p_performance = _safe_logistic(
        cfg.performance_logistic_scale
        * (player_a.performance_score - player_b.performance_score)
    )
    c_performance = min(
        min(
            player_a.performance_innings_total,
            player_b.performance_innings_total,
        )
        / cfg.performance_full_confidence_innings,
        1.0,
    )

    probabilities = {
        "elo": p_elo,
        "career": p_career,
        "season": p_season,
        "recent": p_recent,
        "performance": p_performance,
    }
    confidences = {
        "elo": c_elo,
        "career": c_career,
        "season": c_season,
        "recent": c_recent,
        "performance": c_performance,
    }
    adjusted_weights = {
        name: cfg.category_weights[name] * confidences[name]
        for name in cfg.category_weights
    }
    denominator = sum(adjusted_weights.values())
    if denominator <= 0:
        final_weights = dict(cfg.category_weights)
    else:
        final_weights = {
            name: value / denominator for name, value in adjusted_weights.items()
        }
    p_a = _clip_probability(
        sum(final_weights[name] * probabilities[name] for name in probabilities)
    )

    labels = {
        "elo": "Elo 전력",
        "career": "통산 승률",
        "season": "2026 시즌 승률",
        "recent": "최근 경기 흐름",
        "performance": "세부 경기력",
    }
    factors = sorted(
        (
            {
                "metric": labels[name],
                "advantage": "playerA" if probability >= 0.5 else "playerB",
                "probability": probability,
                "confidence": confidences[name],
                "weight": final_weights[name],
                "impact": abs(final_weights[name] * (probability - 0.5)),
            }
            for name, probability in probabilities.items()
        ),
        key=lambda item: item["impact"],
        reverse=True,
    )

    confidence_score = min(max(denominator, 0.0), 1.0)
    confidence_level = (
        "high" if confidence_score >= 0.8 else "medium" if confidence_score >= 0.55 else "low"
    )
    if p_a >= 0.55:
        display_label = "A_ADVANTAGE"
    elif p_a <= 0.45:
        display_label = "B_ADVANTAGE"
    else:
        display_label = "CLOSE"

    def player_payload(player: PlayerFeatureSnapshot, probability: float) -> dict[str, Any]:
        return {
            "code": player.player_code,
            "name": player.player_name,
            "shortName": player.player_name_short,
            "winProbability": probability,
            "elo": player.elo,
            "career": {
                "matches": player.career_matches,
                "wins": player.career_wins,
                "losses": max(player.career_matches - player.career_wins, 0),
            },
            "season": {
                "matches": player.season_matches,
                "wins": player.season_wins,
                "losses": max(player.season_matches - player.season_wins, 0),
            },
            "recent": {
                "last5Matches": player.last5_matches,
                "last5Wins": player.last5_wins,
                "last10Matches": player.last10_matches,
                "last10Wins": player.last10_wins,
            },
            "performanceScore": player.performance_score,
            "performanceInnings": player.performance_innings_total,
            "metrics": dict(player.metrics),
            "imageUrl": f"/api/v1/players/{player.player_code}/image?league={player.league}",
            "imageIsPlaceholder": player.image_is_placeholder,
        }

    return {
        "modelVersion": "cuecast-prematch-linear-v1",
        "predictionMethod": "confidence-adjusted-linear",
        "league": player_a.league,
        "seasonCode": player_a.season_code,
        "displayLabel": display_label,
        "playerA": player_payload(player_a, p_a),
        "playerB": player_payload(player_b, 1.0 - p_a),
        "componentProbabilities": probabilities,
        "componentConfidences": confidences,
        "baseWeights": dict(cfg.category_weights),
        "finalWeights": final_weights,
        "confidence": {"score": confidence_score, "level": confidence_level},
        "keyFactors": factors[:3],
        "headToHead": dict(head_to_head or {}),
        "headToHeadIncludedInProbability": False,
        "formulaNote": "Elo 30% · 통산 17.5% · 시즌 5% · 최근 2.5% · 세부 경기력 45%, 데이터 신뢰도로 재가중",
    }


class CsvPrematchRepository:
    source = "csv"

    def __init__(self, dataset_root: str | Path):
        self.root = self._resolve_root(Path(dataset_root))
        self.players = self._read_rows("01_players/player_master_2026.csv")
        self.runtime = self._read_rows("02_runtime/player_runtime_state_2026_start.csv")
        self.season = self._read_rows("03_current_season/season_player_state_2026.csv")
        self.h2h = self._read_rows("02_runtime/head_to_head_reference_2026_start.csv")
        self._player_index = {
            (row["league"].upper(), row["player_code"]): row for row in self.players
        }
        self._runtime_index = {
            (row["league"].upper(), _as_int(row["season_code"]), row["player_code"]): row
            for row in self.runtime
        }
        self._season_index = {
            (row["league"].upper(), _as_int(row["season_code"]), row["player_code"]): row
            for row in self.season
        }

    @staticmethod
    def _resolve_root(path: Path) -> Path:
        if (path / "02_runtime" / "player_runtime_state_2026_start.csv").exists():
            return path
        candidates = [
            child
            for child in path.glob("final_dataset*")
            if (child / "02_runtime" / "player_runtime_state_2026_start.csv").exists()
        ]
        if len(candidates) == 1:
            return candidates[0]
        raise PrematchDataError(f"경기 전 예측 데이터셋을 찾을 수 없습니다: {path}")

    def _read_rows(self, relative: str) -> list[dict[str, str]]:
        path = self.root / relative
        if not path.exists():
            raise PrematchDataError(f"필수 데이터 파일이 없습니다: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def list_players(self, league: str, active_only: bool = True) -> list[dict[str, Any]]:
        league = league.upper()
        result = []
        for row in self.players:
            if row["league"].upper() != league:
                continue
            active = _as_bool(row["active_2026_roster"])
            if active_only and not active:
                continue
            result.append(
                {
                    "code": row["player_code"],
                    "name": row["player_name"],
                    "shortName": row["player_name_short"],
                    "league": league,
                    "activeRoster": active,
                    "imageIsPlaceholder": _as_bool(row["image_is_placeholder"]),
                    "imageUrl": f"/api/v1/players/{row['player_code']}/image?league={league}",
                }
            )
        return sorted(result, key=lambda item: (item["name"], item["code"]))

    def get_snapshot(
        self, player_code: str, league: str, season_code: int
    ) -> PlayerFeatureSnapshot:
        key = (league.upper(), season_code, player_code)
        runtime = self._runtime_index.get(key)
        player = self._player_index.get((league.upper(), player_code))
        if runtime is None or player is None:
            raise PrematchDataError(f"선수 스냅샷을 찾을 수 없습니다: {player_code}")
        season = self._season_index.get(key, {})
        season_matches = _as_int(season.get("matches", runtime.get("season_matches")))
        season_wins = _as_int(season.get("wins", runtime.get("season_wins")))
        prior_matches = _as_int(runtime.get("career_matches_prior"))
        prior_wins = _as_int(runtime.get("career_wins_prior"))
        season_innings = _as_float(season.get("innings", runtime.get("season_innings")))
        metrics = {
            "AVG": _as_optional_float(runtime.get("final_avg_start")),
            "TS": _as_optional_float(runtime.get("final_ts_pct_start")),
            "BRS": _as_optional_float(runtime.get("final_brs_pct_start")),
            "5HS": _as_optional_float(runtime.get("final_5hs_pct_start")),
            "HR": _as_optional_float(runtime.get("final_high_run_start")),
        }
        return PlayerFeatureSnapshot(
            league=league.upper(),
            season_code=season_code,
            player_code=player_code,
            player_name=player["player_name"],
            player_name_short=player["player_name_short"],
            active_roster=_as_bool(player["active_2026_roster"]),
            image_is_placeholder=_as_bool(player["image_is_placeholder"]),
            elo=_as_float(season.get("elo_current", runtime.get("elo_start")), 1500.0),
            career_matches=prior_matches + season_matches,
            career_wins=prior_wins + season_wins,
            season_matches=season_matches,
            season_wins=season_wins,
            last5_matches=_as_int(runtime.get("last5_matches_prior")),
            last5_wins=_as_int(runtime.get("last5_wins_prior")),
            last10_matches=_as_int(runtime.get("last10_matches_prior")),
            last10_wins=_as_int(runtime.get("last10_wins_prior")),
            performance_score=_as_float(runtime.get("performance_q_start")),
            performance_innings_total=_as_float(runtime.get("prior_innings")) + season_innings,
            metrics=metrics,
        )

    def get_head_to_head(
        self, player_a_code: str, player_b_code: str, league: str
    ) -> dict[str, Any] | None:
        for row in self.h2h:
            if row["league"].upper() != league.upper():
                continue
            codes = {row["player_a_code"], row["player_b_code"]}
            if codes != {player_a_code, player_b_code}:
                continue
            a_is_stored_a = row["player_a_code"] == player_a_code
            return {
                "matches": _as_int(row["played_matches"]),
                "winsA": _as_int(row["player_a_wins"] if a_is_stored_a else row["player_b_wins"]),
                "winsB": _as_int(row["player_b_wins"] if a_is_stored_a else row["player_a_wins"]),
                "latestWinnerCode": row.get("latest_winner_code") or None,
                "latestWinnerName": row.get("latest_winner_name") or None,
                "lastMatchDatetime": row.get("last_match_datetime") or None,
            }
        return None

    def get_player_image(self, player_code: str, league: str) -> tuple[bytes, str] | None:
        player = self._player_index.get((league.upper(), player_code))
        if player is None:
            return None
        image_path = (self.root / player["image_file"]).resolve()
        if self.root.resolve() not in image_path.parents or not image_path.exists():
            return None
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        return image_path.read_bytes(), mime_type


class PostgresPrematchRepository:
    source = "postgres"

    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError as error:
            raise PrematchDataError("PostgreSQL 연결에는 psycopg2가 필요합니다") from error
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def list_players(self, league: str, active_only: bool = True) -> list[dict[str, Any]]:
        query = """
            SELECT player_code, player_name, player_name_short, league,
                   active_roster, image_is_placeholder
              FROM prematch_players
             WHERE league = %s AND (%s = false OR active_roster = true)
             ORDER BY player_name, player_code
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, (league.upper(), active_only))
            rows = cursor.fetchall()
        return [
            {
                "code": row["player_code"],
                "name": row["player_name"],
                "shortName": row["player_name_short"],
                "league": row["league"],
                "activeRoster": row["active_roster"],
                "imageIsPlaceholder": row["image_is_placeholder"],
                "imageUrl": f"/api/v1/players/{row['player_code']}/image?league={row['league']}",
            }
            for row in rows
        ]

    def get_snapshot(
        self, player_code: str, league: str, season_code: int
    ) -> PlayerFeatureSnapshot:
        query = """
            SELECT f.*, p.player_name, p.player_name_short, p.active_roster,
                   p.image_is_placeholder
              FROM prematch_player_features f
              JOIN prematch_players p USING (league, player_code)
             WHERE f.player_code = %s AND f.league = %s AND f.season_code = %s
             ORDER BY f.snapshot_at DESC
             LIMIT 1
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, (player_code, league.upper(), season_code))
            row = cursor.fetchone()
        if row is None:
            raise PrematchDataError(f"선수 스냅샷을 찾을 수 없습니다: {player_code}")
        return PlayerFeatureSnapshot(
            league=row["league"],
            season_code=row["season_code"],
            player_code=row["player_code"],
            player_name=row["player_name"],
            player_name_short=row["player_name_short"],
            active_roster=row["active_roster"],
            image_is_placeholder=row["image_is_placeholder"],
            elo=float(row["elo"]),
            career_matches=row["career_matches"],
            career_wins=row["career_wins"],
            season_matches=row["season_matches"],
            season_wins=row["season_wins"],
            last5_matches=row["last5_matches"],
            last5_wins=row["last5_wins"],
            last10_matches=row["last10_matches"],
            last10_wins=row["last10_wins"],
            performance_score=float(row["performance_score"]),
            performance_innings_total=float(row["performance_innings_total"]),
            metrics=row["metrics"] or {},
        )

    def get_head_to_head(
        self, player_a_code: str, player_b_code: str, league: str
    ) -> dict[str, Any] | None:
        query = """
            SELECT * FROM prematch_head_to_head
             WHERE league = %s
               AND ((player_a_code = %s AND player_b_code = %s)
                 OR (player_a_code = %s AND player_b_code = %s))
             LIMIT 1
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                query,
                (league.upper(), player_a_code, player_b_code, player_b_code, player_a_code),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        a_is_stored_a = row["player_a_code"] == player_a_code
        return {
            "matches": row["played_matches"],
            "winsA": row["player_a_wins"] if a_is_stored_a else row["player_b_wins"],
            "winsB": row["player_b_wins"] if a_is_stored_a else row["player_a_wins"],
            "latestWinnerCode": row["latest_winner_code"],
            "latestWinnerName": row["latest_winner_name"],
            "lastMatchDatetime": row["last_match_datetime"].isoformat()
            if row["last_match_datetime"]
            else None,
        }

    def get_player_image(self, player_code: str, league: str) -> tuple[bytes, str] | None:
        query = """
            SELECT image_bytes, image_mime_type FROM prematch_players
             WHERE player_code = %s AND league = %s
        """
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(query, (player_code, league.upper()))
            row = cursor.fetchone()
        if row is None or row["image_bytes"] is None:
            return None
        return bytes(row["image_bytes"]), row["image_mime_type"] or "application/octet-stream"


class UnavailablePrematchRepository:
    source = "unavailable"

    def __init__(self, detail: str):
        self.detail = detail

    def _raise(self):
        raise PrematchDataError(self.detail)

    def list_players(self, league: str, active_only: bool = True):
        self._raise()

    def get_snapshot(self, player_code: str, league: str, season_code: int):
        self._raise()

    def get_head_to_head(self, player_a_code: str, player_b_code: str, league: str):
        self._raise()

    def get_player_image(self, player_code: str, league: str):
        self._raise()


class PrematchService:
    def __init__(self, repository: PrematchRepository):
        self.repository = repository

    @property
    def source(self) -> str:
        return self.repository.source

    def list_players(self, league: str, active_only: bool = True) -> list[dict[str, Any]]:
        league = league.upper()
        if league not in {"PBA", "LPBA"}:
            raise PrematchDataError("league는 PBA 또는 LPBA여야 합니다")
        return self.repository.list_players(league, active_only)

    def predict(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        league = str(payload.get("league", "PBA")).upper()
        season_code = _as_int(payload.get("season_code"), 2026)
        player_a_code = str(payload["player_a_code"])
        player_b_code = str(payload["player_b_code"])
        player_a = self.repository.get_snapshot(player_a_code, league, season_code)
        player_b = self.repository.get_snapshot(player_b_code, league, season_code)
        h2h = self.repository.get_head_to_head(player_a_code, player_b_code, league)
        result = predict_from_snapshots(player_a, player_b, head_to_head=h2h)
        result["dataSource"] = self.source
        return result


def create_prematch_service() -> PrematchService:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    dataset_root = os.environ.get("CUECAST_PREMATCH_DATASET_ROOT", "").strip()
    if database_url:
        return PrematchService(PostgresPrematchRepository(database_url))
    if dataset_root:
        try:
            return PrematchService(CsvPrematchRepository(dataset_root))
        except PrematchDataError as error:
            return PrematchService(UnavailablePrematchRepository(str(error)))
    return PrematchService(
        UnavailablePrematchRepository(
            "경기 전 예측 DB가 연결되지 않았습니다. DATABASE_URL 또는 "
            "CUECAST_PREMATCH_DATASET_ROOT를 설정해 주세요."
        )
    )
