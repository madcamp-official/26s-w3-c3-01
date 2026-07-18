from __future__ import annotations

from math import comb, exp, log


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
                {
                    "winner": "a",
                    "winnerSets": sets_to_win,
                    "loserSets": losses,
                    "probability": a_probability,
                },
                {
                    "winner": "b",
                    "winnerSets": sets_to_win,
                    "loserSets": losses,
                    "probability": b_probability,
                },
            )
        )
    return outcomes


def predict_match_probability(
    player_a: str,
    player_b: str,
    avg_a: float,
    avg_b: float,
    *,
    sets_to_win: int = 4,
) -> dict[str, object]:
    """Estimate a set-series result from the two players' scoring averages."""
    if not player_a.strip() or not player_b.strip():
        raise ValueError("두 선수 이름이 모두 필요합니다")
    if not 0.1 <= avg_a <= 5.0 or not 0.1 <= avg_b <= 5.0:
        raise ValueError("AVG는 0.1~5.0 사이여야 합니다")
    if not 1 <= sets_to_win <= 7:
        raise ValueError("승리 필요 세트는 1~7 사이여야 합니다")

    # A conservative Bradley-Terry prior. This can later be calibrated against
    # the match database without changing the API or UI contract.
    log_odds = 2.4 * log(avg_a / avg_b)
    set_probability_a = 1.0 / (1.0 + exp(-log_odds))
    outcomes = _series_outcomes(set_probability_a, sets_to_win)
    match_probability_a = sum(
        float(outcome["probability"])
        for outcome in outcomes
        if outcome["winner"] == "a"
    )
    likely_score = max(outcomes, key=lambda outcome: float(outcome["probability"]))

    return {
        "modelVersion": "avg-bradley-terry-v1",
        "setsToWin": sets_to_win,
        "playerA": {
            "name": player_a.strip(),
            "avg": avg_a,
            "setProbability": set_probability_a,
            "winProbability": match_probability_a,
        },
        "playerB": {
            "name": player_b.strip(),
            "avg": avg_b,
            "setProbability": 1.0 - set_probability_a,
            "winProbability": 1.0 - match_probability_a,
        },
        "likelyScore": likely_score,
        "inputs": ["player_avg", "sets_to_win"],
        "provisional": True,
    }
