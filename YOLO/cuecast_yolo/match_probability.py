from __future__ import annotations

from math import comb, exp, log
from typing import Mapping


LEAGUE_PRIORS = {
    "match_rate": (0.5, 12.0),
    "set_rate": (0.5, 20.0),
    "shot_rate": (0.42, 60.0),
    "break_rate": (0.38, 15.0),
    "bank_rate": (0.18, 20.0),
    "long_run_rate": (0.12, 20.0),
    "shootout_rate": (0.5, 10.0),
}


def _series_outcomes(set_probability: float, sets_to_win: int) -> list[dict[str, object]]:
    outcomes: list[dict[str, object]] = []
    for losses in range(sets_to_win):
        a_probability = (
            comb(sets_to_win + losses - 1, losses)
            * set_probability**sets_to_win
            * (1.0 - set_probability) ** losses
        )
        b_probability = (
            comb(sets_to_win + losses - 1, losses)
            * (1.0 - set_probability) ** sets_to_win
            * set_probability**losses
        )
        outcomes.extend(
            (
                {"winner": "a", "winnerSets": sets_to_win, "loserSets": losses, "probability": a_probability},
                {"winner": "b", "winnerSets": sets_to_win, "loserSets": losses, "probability": b_probability},
            )
        )
    return outcomes


def _number(stats: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = stats.get(key, default)
    try:
        result = float(value.strip().removesuffix("%")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, result)


def _percentage(stats: Mapping[str, object], key: str) -> float | None:
    if key not in stats or stats[key] in (None, ""):
        return None
    value = _number(stats, key)
    if value > 1.0:
        value /= 100.0
    return min(value, 1.0)


def _smoothed_rate(successes: float, attempts: float, prior_key: str) -> float:
    prior, strength = LEAGUE_PRIORS[prior_key]
    if attempts <= 0:
        return prior
    successes = min(max(successes, 0.0), attempts)
    return (successes + prior * strength) / (attempts + strength)


def _rate_from_counts_or_percent(
    stats: Mapping[str, object],
    success_key: str,
    failure_or_attempt_key: str,
    percent_key: str,
    prior_key: str,
    *,
    second_is_failure: bool = False,
) -> tuple[float, float]:
    successes = _number(stats, success_key)
    second = _number(stats, failure_or_attempt_key)
    attempts = successes + second if second_is_failure else second
    if success_key in stats and failure_or_attempt_key in stats and attempts > 0:
        return _smoothed_rate(successes, attempts, prior_key), attempts
    percentage = _percentage(stats, percent_key)
    if percentage is not None:
        _, strength = LEAGUE_PRIORS[prior_key]
        return (percentage * strength + LEAGUE_PRIORS[prior_key][0] * strength) / (2 * strength), strength
    return LEAGUE_PRIORS[prior_key][0], 0.0


def _player_features(avg: float, stats: Mapping[str, object] | None) -> dict[str, float]:
    source = stats or {}
    match_rate, matches = _rate_from_counts_or_percent(source, "W", "L", "W%", "match_rate", second_is_failure=True)
    set_rate, sets = _rate_from_counts_or_percent(source, "SW", "SL", "SW%", "set_rate", second_is_failure=True)
    shot_rate, attempts = _rate_from_counts_or_percent(source, "TS", "TA", "TS%", "shot_rate")
    break_rate, break_samples = _rate_from_counts_or_percent(source, "BRS", "BRTA", "BRS%", "break_rate")
    bank_rate, bank_samples = _rate_from_counts_or_percent(source, "BS", "TS", "BS%", "bank_rate")
    long_run_rate, long_samples = _rate_from_counts_or_percent(source, "5HS", "Inn", "5HS%", "long_run_rate")
    shootout_rate, shootouts = _rate_from_counts_or_percent(source, "PW", "PL", "PWR", "shootout_rate", second_is_failure=True)
    career_rate = _percentage(source, "career_win_rate_before")
    season_rate = _percentage(source, "season_win_rate_before")
    last5_rate = _percentage(source, "last5_win_rate_before")
    last10_rate = _percentage(source, "last10_win_rate_before")
    h2h_rate = _percentage(source, "h2h_win_rate_before")
    return {
        "avg": avg,
        "match_rate": career_rate if career_rate is not None else match_rate,
        "season_rate": season_rate if season_rate is not None else match_rate,
        "last5_rate": last5_rate if last5_rate is not None else (season_rate if season_rate is not None else match_rate),
        "last10_rate": last10_rate if last10_rate is not None else (season_rate if season_rate is not None else match_rate),
        "h2h_rate": h2h_rate if h2h_rate is not None else 0.5,
        "set_rate": set_rate,
        "shot_rate": shot_rate,
        "break_rate": break_rate,
        "bank_rate": bank_rate,
        "long_run_rate": long_run_rate,
        "shootout_rate": shootout_rate,
        "matches": matches,
        "sets": sets,
        "attempts": attempts,
        "aux_samples": break_samples + bank_samples + long_samples + shootouts,
    }


def _confidence(features_a: Mapping[str, float], features_b: Mapping[str, float]) -> dict[str, object]:
    matches = min(features_a["matches"], features_b["matches"])
    sets = min(features_a["sets"], features_b["sets"])
    attempts = min(features_a["attempts"], features_b["attempts"])
    score = 0.2 + 0.3 * min(matches / 30.0, 1.0) + 0.2 * min(sets / 80.0, 1.0) + 0.3 * min(attempts / 500.0, 1.0)
    level = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return {"score": round(score, 3), "level": level, "minimumMatches": int(matches), "minimumAttempts": int(attempts)}


def predict_match_probability(
    player_a: str,
    player_b: str,
    avg_a: float,
    avg_b: float,
    *,
    sets_to_win: int = 4,
    stats_a: Mapping[str, object] | None = None,
    stats_b: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """MVP pregame forecast using smoothed player statistics and exact set-series odds."""
    if not player_a.strip() or not player_b.strip():
        raise ValueError("두 선수 이름이 모두 필요합니다")
    if not 0.1 <= avg_a <= 5.0 or not 0.1 <= avg_b <= 5.0:
        raise ValueError("AVG는 0.1~5.0 사이여야 합니다")
    if not 1 <= sets_to_win <= 7:
        raise ValueError("승리 필요 세트는 1~7 사이여야 합니다")

    a = _player_features(avg_a, stats_a)
    b = _player_features(avg_b, stats_b)
    contributions = [
        ("AVG", 1.65 * log(a["avg"] / b["avg"])),
        ("누적 승률", 0.65 * (a["match_rate"] - b["match_rate"])),
        ("시즌 승률", 0.55 * (a["season_rate"] - b["season_rate"])),
        ("최근 5경기", 0.25 * (a["last5_rate"] - b["last5_rate"])),
        ("최근 10경기", 0.35 * (a["last10_rate"] - b["last10_rate"])),
        ("상대 전적", 0.20 * (a["h2h_rate"] - b["h2h_rate"])),
        ("세트 승률", 1.05 * (a["set_rate"] - b["set_rate"])),
        ("공격 성공률", 1.80 * (a["shot_rate"] - b["shot_rate"])),
        ("초구 성공률", 0.45 * (a["break_rate"] - b["break_rate"])),
        ("뱅크샷 비율", 0.20 * (a["bank_rate"] - b["bank_rate"])),
        ("장타율", 0.25 * (a["long_run_rate"] - b["long_run_rate"])),
        ("승부치기 승률", 0.15 * (a["shootout_rate"] - b["shootout_rate"])),
    ]
    log_odds = sum(value for _, value in contributions)
    set_probability_a = 1.0 / (1.0 + exp(-log_odds))
    outcomes = _series_outcomes(set_probability_a, sets_to_win)
    match_probability_a = sum(float(outcome["probability"]) for outcome in outcomes if outcome["winner"] == "a")
    likely_score = max(outcomes, key=lambda outcome: float(outcome["probability"]))
    key_factors = [
        {"metric": metric, "advantage": "playerA" if value > 0 else "playerB", "impact": round(abs(value), 4)}
        for metric, value in sorted(contributions, key=lambda item: abs(item[1]), reverse=True)
        if abs(value) >= 0.01
    ][:3]

    return {
        "modelVersion": "pregame-statistical-mvp-v1",
        "setsToWin": sets_to_win,
        "playerA": {"name": player_a.strip(), "avg": avg_a, "setProbability": set_probability_a, "winProbability": match_probability_a},
        "playerB": {"name": player_b.strip(), "avg": avg_b, "setProbability": 1.0 - set_probability_a, "winProbability": 1.0 - match_probability_a},
        "likelyScore": likely_score,
        "confidence": _confidence(a, b),
        "keyFactors": key_factors,
        "inputs": ["AVG", "W/L", "SW/SL", "TS/TA", "BRS%", "BS%", "5HS%", "PWR", "career/season/last5/last10/h2h_before", "sets_to_win"],
        "provisional": True,
    }
