from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log


EPSILON = 1e-9


def _clamp_probability(value: float) -> float:
    return min(1.0 - EPSILON, max(EPSILON, float(value)))


def _logit(value: float) -> float:
    probability = _clamp_probability(value)
    return log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        negative = exp(-value)
        return 1.0 / (1.0 + negative)
    positive = exp(value)
    return positive / (1.0 + positive)


def avg_to_shot_probability(avg: float) -> float:
    """Convert expected points per inning to a geometric next-point chance."""
    if avg <= 0:
        raise ValueError("AVG는 0보다 커야 합니다")
    return float(avg) / (1.0 + float(avg))


@dataclass(frozen=True)
class ScoreDp:
    target_score: int
    attacker_a: tuple[tuple[float, ...], ...]
    attacker_b: tuple[tuple[float, ...], ...]

    def probability(self, score_a: int, score_b: int, attacker: str) -> float:
        if score_a >= self.target_score:
            return 1.0
        if score_b >= self.target_score:
            return 0.0
        table = self.attacker_a if attacker == "a" else self.attacker_b
        return table[score_a][score_b]


def build_score_dp(target_score: int, p_a: float, p_b: float) -> ScoreDp:
    """Build the PDF's closed-form score/possession dynamic-programming table."""
    if target_score < 1:
        raise ValueError("target_score는 1 이상이어야 합니다")
    p_a = _clamp_probability(p_a)
    p_b = _clamp_probability(p_b)
    denominator = p_a + p_b - p_a * p_b
    attacker_a = [[0.0] * (target_score + 1) for _ in range(target_score + 1)]
    attacker_b = [[0.0] * (target_score + 1) for _ in range(target_score + 1)]
    for score_b in range(target_score + 1):
        attacker_a[target_score][score_b] = 1.0
        attacker_b[target_score][score_b] = 1.0
    for score_a in range(target_score):
        attacker_a[score_a][target_score] = 0.0
        attacker_b[score_a][target_score] = 0.0
    for score_a in range(target_score - 1, -1, -1):
        for score_b in range(target_score - 1, -1, -1):
            next_a = attacker_a[score_a + 1][score_b]
            next_b = attacker_b[score_a][score_b + 1]
            attacker_a[score_a][score_b] = (
                p_a * next_a + (1.0 - p_a) * p_b * next_b
            ) / denominator
            attacker_b[score_a][score_b] = (
                p_b * next_b + (1.0 - p_b) * p_a * next_a
            ) / denominator
    return ScoreDp(
        target_score=target_score,
        attacker_a=tuple(tuple(row) for row in attacker_a),
        attacker_b=tuple(tuple(row) for row in attacker_b),
    )


def calibrate_base_probabilities(
    raw_a: float,
    raw_b: float,
    prematch_probability_a: float,
    target_score: int,
    starting_player: str,
    sets_to_win: int = 1,
) -> tuple[float, float, float, float]:
    """Find the shared log-odds shift that preserves the pre-match match odds."""
    if starting_player not in ("a", "b"):
        raise ValueError("starting_player는 a 또는 b여야 합니다")
    target = _clamp_probability(prematch_probability_a)
    raw_logit_a = _logit(raw_a)
    raw_logit_b = _logit(raw_b)

    def evaluate(delta: float) -> tuple[float, float, float]:
        p_a = _sigmoid(raw_logit_a + delta / 2.0)
        p_b = _sigmoid(raw_logit_b - delta / 2.0)
        dp = build_score_dp(target_score, p_a, p_b)
        first_set_probability = dp.probability(0, 0, starting_player)
        if sets_to_win == 1:
            calibrated = first_set_probability
        else:
            set_probabilities = {
                starter: dp.probability(0, 0, starter)
                for starter in ("a", "b")
            }
            next_starter = "b" if starting_player == "a" else "a"
            calibrated = _series_probability(
                0,
                0,
                sets_to_win,
                first_set_probability,
                next_starter,
                set_probabilities,
            )
        return p_a, p_b, calibrated

    low, high = -24.0, 24.0
    p_a = p_b = calibrated = 0.5
    for _ in range(80):
        middle = (low + high) / 2.0
        p_a, p_b, calibrated = evaluate(middle)
        if calibrated < target:
            low = middle
        else:
            high = middle
    error = calibrated - target
    return p_a, p_b, (low + high) / 2.0, error


def _future_probability(base: float, hits: int, misses: int, strength: float) -> float:
    if hits < 0 or misses < 0:
        raise ValueError("hits와 misses는 0 이상이어야 합니다")
    if strength <= 0:
        raise ValueError("prior_strength는 0보다 커야 합니다")
    return (strength * base + hits) / (strength + hits + misses)


def _series_probability(
    sets_won_a: int,
    sets_won_b: int,
    sets_to_win: int,
    first_set_probability_a: float,
    next_starting_player: str,
    future_set_probability: dict[str, float],
) -> float:
    if sets_won_a >= sets_to_win:
        return 1.0
    if sets_won_b >= sets_to_win:
        return 0.0

    @lru_cache(maxsize=None)
    def future(a_sets: int, b_sets: int, starter: str) -> float:
        if a_sets >= sets_to_win:
            return 1.0
        if b_sets >= sets_to_win:
            return 0.0
        set_probability = future_set_probability[starter]
        following = "b" if starter == "a" else "a"
        return set_probability * future(a_sets + 1, b_sets, following) + (
            1.0 - set_probability
        ) * future(a_sets, b_sets + 1, following)

    return first_set_probability_a * future(
        sets_won_a + 1, sets_won_b, next_starting_player
    ) + (1.0 - first_set_probability_a) * future(
        sets_won_a, sets_won_b + 1, next_starting_player
    )


def predict_live_match_probability(
    *,
    prematch_probability_a: float,
    final_avg_a: float,
    final_avg_b: float,
    score_a: int,
    score_b: int,
    target_score: int,
    starting_player: str,
    current_player: str,
    current_shot_probability: float,
    live_hits_a: int = 0,
    live_misses_a: int = 0,
    live_hits_b: int = 0,
    live_misses_b: int = 0,
    prior_strength: float = 12.0,
    sets_won_a: int = 0,
    sets_won_b: int = 0,
    sets_to_win: int = 1,
    previous_probability_a: float | None = None,
) -> dict[str, object]:
    """Calculate current set and whole-match odds from the supplied live state."""
    if current_player not in ("a", "b"):
        raise ValueError("current_player는 a 또는 b여야 합니다")
    if starting_player not in ("a", "b"):
        raise ValueError("starting_player는 a 또는 b여야 합니다")
    if score_a < 0 or score_b < 0:
        raise ValueError("점수는 0 이상이어야 합니다")
    if not 1 <= sets_to_win <= 7:
        raise ValueError("sets_to_win은 1~7 사이여야 합니다")
    if sets_won_a < 0 or sets_won_b < 0:
        raise ValueError("세트 스코어는 0 이상이어야 합니다")
    q = min(1.0, max(0.0, float(current_shot_probability)))
    raw_a = avg_to_shot_probability(final_avg_a)
    raw_b = avg_to_shot_probability(final_avg_b)
    base_a, base_b, delta, calibration_error = calibrate_base_probabilities(
        raw_a,
        raw_b,
        prematch_probability_a,
        target_score,
        starting_player,
        sets_to_win,
    )
    future_a = _future_probability(base_a, live_hits_a, live_misses_a, prior_strength)
    future_b = _future_probability(base_b, live_hits_b, live_misses_b, prior_strength)
    dp = build_score_dp(target_score, future_a, future_b)

    if score_a >= target_score:
        success_probability_a = failure_probability_a = set_probability_a = 1.0
    elif score_b >= target_score:
        success_probability_a = failure_probability_a = set_probability_a = 0.0
    elif current_player == "a":
        success_probability_a = dp.probability(score_a + 1, score_b, "a")
        failure_probability_a = dp.probability(score_a, score_b, "b")
        set_probability_a = (
            q * success_probability_a + (1.0 - q) * failure_probability_a
        )
    else:
        success_probability_a = dp.probability(score_a, score_b + 1, "b")
        failure_probability_a = dp.probability(score_a, score_b, "a")
        set_probability_a = (
            q * success_probability_a + (1.0 - q) * failure_probability_a
        )

    state_probability_a = dp.probability(score_a, score_b, current_player)
    future_set_probability = {
        starter: build_score_dp(target_score, future_a, future_b).probability(
            0, 0, starter
        )
        for starter in ("a", "b")
    }
    next_starter = "b" if starting_player == "a" else "a"
    match_probability_a = _series_probability(
        sets_won_a,
        sets_won_b,
        sets_to_win,
        set_probability_a,
        next_starter,
        future_set_probability,
    )
    prematch_match_probability_a = _clamp_probability(prematch_probability_a)
    prematch_set_probability_a = build_score_dp(
        target_score, base_a, base_b
    ).probability(0, 0, starting_player)
    previous = (
        float(previous_probability_a)
        if previous_probability_a is not None
        else prematch_match_probability_a
    )
    return {
        "modelVersion": "live-match-dp-v1",
        "setWinProbabilityA": set_probability_a,
        "setWinProbabilityB": 1.0 - set_probability_a,
        "matchWinProbabilityA": match_probability_a,
        "matchWinProbabilityB": 1.0 - match_probability_a,
        "changeFromPreviousA": match_probability_a - previous,
        "changeFromPrematchA": match_probability_a - prematch_match_probability_a,
        "scoreAndPossessionProbabilityA": state_probability_a,
        "currentShotImpactA": set_probability_a - state_probability_a,
        "winProbabilityAIfCurrentShotSucceeds": success_probability_a,
        "winProbabilityAIfCurrentShotFails": failure_probability_a,
        "shotSwing": abs(success_probability_a - failure_probability_a),
        "calibratedBaseShotProbabilityA": base_a,
        "calibratedBaseShotProbabilityB": base_b,
        "liveFutureShotProbabilityA": future_a,
        "liveFutureShotProbabilityB": future_b,
        "calibrationDelta": delta,
        "calibrationError": calibration_error,
        "prematchSetProbabilityA": prematch_set_probability_a,
        "prematchMatchProbabilityA": prematch_match_probability_a,
        "inputs": {
            "scoreA": score_a,
            "scoreB": score_b,
            "targetScore": target_score,
            "startingPlayer": starting_player,
            "currentPlayer": current_player,
            "currentShotProbability": q,
            "setsWonA": sets_won_a,
            "setsWonB": sets_won_b,
            "setsToWin": sets_to_win,
        },
    }
