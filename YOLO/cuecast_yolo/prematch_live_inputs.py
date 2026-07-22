from __future__ import annotations

from difflib import SequenceMatcher
from threading import Lock
from time import monotonic
import unicodedata

from .prematch_probability import PrematchDataError, PrematchService


def normalize_broadcast_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def broadcast_name_similarity(ocr_name: str, db_name: str) -> float:
    """Return a character-level score suitable for short Korean broadcast names."""
    target = normalize_broadcast_name(ocr_name)
    candidate = normalize_broadcast_name(db_name)
    if not target or not candidate:
        return 0.0
    if target == candidate:
        return 1.0
    if candidate in target:
        return max(0.9, len(candidate) / len(target))
    if target in candidate:
        return max(0.82, len(target) / len(candidate))

    def windowed_ratio(left: str, right: str) -> float:
        global_score = SequenceMatcher(None, left, right).ratio()
        if len(left) <= len(right):
            return global_score
        window_score = max(
            SequenceMatcher(None, left[index : index + len(right)], right).ratio()
            for index in range(len(left) - len(right) + 1)
        )
        return max(global_score, window_score)

    character_score = windowed_ratio(target, candidate)
    jamo_score = windowed_ratio(
        unicodedata.normalize("NFD", target),
        unicodedata.normalize("NFD", candidate),
    )
    return max(character_score, jamo_score)


class LockedScoreboardPlayerMatcher:
    """Keep the first confident DB match until the scoreboard is explicitly reset."""

    def __init__(self, provider: "PrematchLiveInputProvider") -> None:
        self.provider = provider
        self._lock = Lock()
        self._generation = 0
        self._fixed: dict[str, dict[str, object]] = {}

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            self._fixed = {}

    def match(self, scoreboard: dict[str, object]) -> dict[str, object]:
        with self._lock:
            generation = self._generation
            fixed_before = dict(self._fixed)
        if len(fixed_before) == 2:
            return self._apply_fixed(scoreboard, fixed_before)

        resolved = self.provider.match_scoreboard_players(scoreboard)
        with self._lock:
            if generation != self._generation:
                return dict(scoreboard)
            for number in ("1", "2"):
                field = f"player{number}Name"
                if field in self._fixed:
                    continue
                if resolved.get(f"player{number}NameSimilarity") is None:
                    continue
                self._fixed[field] = {
                    key: value
                    for key, value in resolved.items()
                    if key == field or key.startswith(f"player{number}")
                }
            fixed = dict(self._fixed)
        return self._apply_fixed(resolved, fixed)

    @staticmethod
    def _apply_fixed(
        scoreboard: dict[str, object], fixed: dict[str, dict[str, object]]
    ) -> dict[str, object]:
        result = dict(scoreboard)
        for payload in fixed.values():
            result.update(payload)
        return result


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

    def match_scoreboard_players(
        self,
        scoreboard: dict[str, object],
        *,
        minimum_similarity: float = 0.5,
    ) -> dict[str, object]:
        """Replace OCR names with the closest distinct DB players from one league."""
        fields = ("player1Name", "player2Name")
        raw_names = {
            field: str(scoreboard.get(field) or "").strip()
            for field in fields
            if str(scoreboard.get(field) or "").strip()
        }
        if not raw_names:
            return dict(scoreboard)

        best: tuple[float, str, dict[str, tuple[dict[str, object], float]]] | None = None
        for league in ("PBA", "LPBA"):
            players = self._players(league)
            rankings: dict[str, list[tuple[dict[str, object], float]]] = {}
            for field, raw_name in raw_names.items():
                rankings[field] = sorted(
                    (
                        (player, broadcast_name_similarity(raw_name, str(player["name"])))
                        for player in players
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            if any(not ranked for ranked in rankings.values()):
                continue

            choices: list[dict[str, tuple[dict[str, object], float]]]
            if len(rankings) == 1:
                field = next(iter(rankings))
                choices = [{field: rankings[field][0]}]
            else:
                first, second = fields
                choices = [
                    {first: left, second: right}
                    for left in rankings[first]
                    for right in rankings[second]
                    if str(left[0]["code"]) != str(right[0]["code"])
                ]
            for choice in choices:
                score = sum(item[1] for item in choice.values()) / len(choice)
                if best is None or score > best[0]:
                    best = (score, league, choice)

        matched = dict(scoreboard)
        if best is None:
            return matched
        _, league, choices = best
        for field, (player, similarity) in choices.items():
            if similarity < minimum_similarity:
                continue
            number = "1" if field == "player1Name" else "2"
            matched[f"player{number}OcrName"] = raw_names[field]
            matched[f"player{number}NameSimilarity"] = round(similarity, 3)
            matched[f"player{number}League"] = league
            matched[field] = str(player["name"])
        return matched

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
