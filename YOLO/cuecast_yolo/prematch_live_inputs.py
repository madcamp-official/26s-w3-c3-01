from __future__ import annotations

from time import monotonic
import unicodedata

from .prematch_probability import PrematchDataError, PrematchService


def normalize_broadcast_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


class PrematchLiveInputProvider:
    """Adapt the DB-backed pre-match model output for the live DP engine."""

    def __init__(
        self,
        service: PrematchService,
        *,
        season_code: int = 2026,
        cache_seconds: float = 60.0,
    ) -> None:
        self.service = service
        self.season_code = season_code
        self.cache_seconds = cache_seconds
        self._players_cache: dict[str, tuple[float, list[dict[str, object]]]] = {}
        self._prediction_cache: dict[
            tuple[str, str, str, int], tuple[float, dict[str, object]]
        ] = {}

    def _players(self, league: str) -> list[dict[str, object]]:
        cached = self._players_cache.get(league)
        if cached and monotonic() - cached[0] < self.cache_seconds:
            return cached[1]
        players = self.service.list_players(league, active_only=False)
        self._players_cache[league] = (monotonic(), players)
        return players

    def _resolve_pair(
        self, player_a: str, player_b: str
    ) -> tuple[str, dict[str, object], dict[str, object]]:
        target_a = normalize_broadcast_name(player_a)
        target_b = normalize_broadcast_name(player_b)
        for league in ("PBA", "LPBA"):
            index = {
                normalize_broadcast_name(str(player["name"])): player
                for player in self._players(league)
            }

            def resolve(target: str) -> dict[str, object] | None:
                if target in index:
                    return index[target]
                candidates = [
                    player
                    for normalized_name, player in index.items()
                    if normalized_name and normalized_name in target
                ]
                return candidates[0] if len(candidates) == 1 else None

            resolved_a = resolve(target_a)
            resolved_b = resolve(target_b)
            if resolved_a is not None and resolved_b is not None:
                return league, resolved_a, resolved_b
        raise PrematchDataError(
            f"경기 전 DB에서 두 선수를 찾을 수 없습니다: {player_a}, {player_b}"
        )

    def fetch(
        self,
        player_a: str,
        player_b: str,
        *,
        set_number: int,
        format_key: str = "pba-default",
    ) -> dict[str, object]:
        league, resolved_a, resolved_b = self._resolve_pair(player_a, player_b)
        cache_key = (
            league,
            str(resolved_a["code"]),
            str(resolved_b["code"]),
            self.season_code,
        )
        cached = self._prediction_cache.get(cache_key)
        if cached and monotonic() - cached[0] < self.cache_seconds:
            prediction = dict(cached[1])
        else:
            prediction = self.service.predict(
                {
                    "league": league,
                    "season_code": self.season_code,
                    "player_a_code": cache_key[1],
                    "player_b_code": cache_key[2],
                }
            )
            self._prediction_cache[cache_key] = (monotonic(), dict(prediction))
        prediction_a = prediction.get("playerA")
        prediction_b = prediction.get("playerB")
        if not isinstance(prediction_a, dict) or not isinstance(prediction_b, dict):
            raise PrematchDataError("경기 전 승률 응답에 선수 정보가 없습니다")
        metrics_a = prediction_a.get("metrics")
        metrics_b = prediction_b.get("metrics")
        avg_a = metrics_a.get("AVG") if isinstance(metrics_a, dict) else None
        avg_b = metrics_b.get("AVG") if isinstance(metrics_b, dict) else None
        if avg_a is None or float(avg_a) <= 0:
            raise PrematchDataError(f"{prediction_a.get('name', player_a)} AVG가 없습니다")
        if avg_b is None or float(avg_b) <= 0:
            raise PrematchDataError(f"{prediction_b.get('name', player_b)} AVG가 없습니다")
        sets_to_win = 4
        deciding_set_number = sets_to_win * 2 - 1
        return {
            "playerA": {**prediction_a, "avgFinal": float(avg_a)},
            "playerB": {**prediction_b, "avgFinal": float(avg_b)},
            "format": {
                "key": format_key,
                "targetScore": 11 if set_number >= deciding_set_number else 15,
                "regularTargetScore": 15,
                "decidingSetTargetScore": 11,
                "setsToWin": sets_to_win,
                "source": "pba_rules",
            },
            "prematchProbabilityA": float(prediction_a["winProbability"]),
            "prematchSource": str(
                prediction.get("modelVersion", "cuecast-prematch-linear-v1")
            ),
            "dataSource": prediction.get("dataSource", self.service.source),
            "prematch": prediction,
            "league": league,
        }
