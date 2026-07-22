from __future__ import annotations

import unittest

from cuecast_yolo.live_match_probability import (
    avg_to_shot_probability,
    build_score_dp,
    calibrate_base_probabilities,
    predict_live_match_probability,
)


class LiveMatchProbabilityTest(unittest.TestCase):
    def base_inputs(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "prematch_probability_a": 0.5,
            "final_avg_a": 1.5,
            "final_avg_b": 1.5,
            "score_a": 4,
            "score_b": 4,
            "target_score": 15,
            "starting_player": "a",
            "current_player": "a",
            "current_shot_probability": 0.5,
            "sets_won_a": 0,
            "sets_won_b": 0,
            "sets_to_win": 4,
        }
        values.update(overrides)
        return values

    def test_avg_is_converted_with_geometric_expectation(self) -> None:
        self.assertAlmostEqual(avg_to_shot_probability(1.0), 0.5)
        self.assertAlmostEqual(avg_to_shot_probability(1.5), 0.6)

    def test_calibration_preserves_prematch_set_probability(self) -> None:
        raw_a = avg_to_shot_probability(1.7)
        raw_b = avg_to_shot_probability(1.3)
        p_a, p_b, _delta, error = calibrate_base_probabilities(
            raw_a, raw_b, 0.63, 15, "a"
        )
        dp = build_score_dp(15, p_a, p_b)
        self.assertAlmostEqual(dp.probability(0, 0, "a"), 0.63, places=7)
        self.assertAlmostEqual(error, 0.0, places=7)

    def test_series_calibration_preserves_remote_prematch_probability(self) -> None:
        first = predict_live_match_probability(
            **self.base_inputs(
                prematch_probability_a=0.67,
                score_a=0,
                score_b=0,
                current_shot_probability=0.5,
            )
        )
        calibrated_q = first["calibratedBaseShotProbabilityA"]
        result = predict_live_match_probability(
            **self.base_inputs(
                prematch_probability_a=0.67,
                score_a=0,
                score_b=0,
                current_shot_probability=calibrated_q,
            )
        )
        self.assertAlmostEqual(result["matchWinProbabilityA"], 0.67, places=6)
        self.assertAlmostEqual(result["calibrationError"], 0.0, places=7)

    def test_terminal_scores_are_exact(self) -> None:
        won = predict_live_match_probability(
            **self.base_inputs(score_a=15, score_b=9)
        )
        lost = predict_live_match_probability(
            **self.base_inputs(score_a=9, score_b=15)
        )
        self.assertEqual(won["setWinProbabilityA"], 1.0)
        self.assertEqual(lost["setWinProbabilityA"], 0.0)

    def test_player_probabilities_sum_to_one(self) -> None:
        result = predict_live_match_probability(**self.base_inputs())
        self.assertAlmostEqual(
            result["setWinProbabilityA"] + result["setWinProbabilityB"], 1.0
        )
        self.assertAlmostEqual(
            result["matchWinProbabilityA"] + result["matchWinProbabilityB"],
            1.0,
        )

    def test_a_score_increase_cannot_reduce_a_probability(self) -> None:
        before = predict_live_match_probability(**self.base_inputs(score_a=4))
        after = predict_live_match_probability(**self.base_inputs(score_a=5))
        self.assertGreaterEqual(
            after["setWinProbabilityA"], before["setWinProbabilityA"]
        )

    def test_shot_probability_is_monotonic_for_current_attacker(self) -> None:
        a_low = predict_live_match_probability(
            **self.base_inputs(current_player="a", current_shot_probability=0.2)
        )
        a_high = predict_live_match_probability(
            **self.base_inputs(current_player="a", current_shot_probability=0.8)
        )
        b_low = predict_live_match_probability(
            **self.base_inputs(current_player="b", current_shot_probability=0.2)
        )
        b_high = predict_live_match_probability(
            **self.base_inputs(current_player="b", current_shot_probability=0.8)
        )
        self.assertGreater(
            a_high["setWinProbabilityA"], a_low["setWinProbabilityA"]
        )
        self.assertLess(
            b_high["setWinProbabilityA"], b_low["setWinProbabilityA"]
        )

    def test_current_shot_conditions_follow_score_and_possession_transition(self) -> None:
        result = predict_live_match_probability(**self.base_inputs(current_player="a"))
        p_a = result["liveFutureShotProbabilityA"]
        p_b = result["liveFutureShotProbabilityB"]
        dp = build_score_dp(15, p_a, p_b)
        self.assertAlmostEqual(
            result["winProbabilityAIfCurrentShotSucceeds"],
            dp.probability(5, 4, "a"),
        )
        self.assertAlmostEqual(
            result["winProbabilityAIfCurrentShotFails"],
            dp.probability(4, 4, "b"),
        )

    def test_set_score_is_applied_to_whole_match(self) -> None:
        tied = predict_live_match_probability(**self.base_inputs())
        leading = predict_live_match_probability(
            **self.base_inputs(sets_won_a=2, sets_won_b=0)
        )
        self.assertGreater(
            leading["matchWinProbabilityA"], tied["matchWinProbabilityA"]
        )


if __name__ == "__main__":
    unittest.main()
