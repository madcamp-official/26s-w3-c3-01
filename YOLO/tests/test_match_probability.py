from __future__ import annotations

import unittest

from cuecast_yolo.match_probability import predict_match_probability


class MatchProbabilityTest(unittest.TestCase):
    def test_equal_averages_produce_even_match(self) -> None:
        result = predict_match_probability("A", "B", 1.5, 1.5, sets_to_win=4)
        self.assertAlmostEqual(result["playerA"]["winProbability"], 0.5)
        self.assertAlmostEqual(result["playerB"]["winProbability"], 0.5)

    def test_higher_average_is_favorite(self) -> None:
        result = predict_match_probability("A", "B", 1.8, 1.3, sets_to_win=4)
        self.assertGreater(result["playerA"]["winProbability"], 0.5)
        self.assertEqual(result["likelyScore"]["winner"], "a")

    def test_probabilities_sum_to_one(self) -> None:
        result = predict_match_probability("A", "B", 1.62, 1.49, sets_to_win=3)
        total = (
            result["playerA"]["winProbability"]
            + result["playerB"]["winProbability"]
        )
        self.assertAlmostEqual(total, 1.0)

    def test_full_stats_influence_forecast(self) -> None:
        result = predict_match_probability(
            "A",
            "B",
            1.5,
            1.5,
            sets_to_win=3,
            stats_a={"W": 30, "L": 10, "SW": 80, "SL": 35, "TS%": 48.0, "BRS%": 45.0},
            stats_b={"W": 10, "L": 30, "SW": 35, "SL": 80, "TS%": 37.0, "BRS%": 30.0},
        )
        self.assertGreater(result["playerA"]["winProbability"], 0.5)
        self.assertEqual(result["modelVersion"], "pregame-statistical-mvp-v1")
        self.assertTrue(result["keyFactors"])
        self.assertEqual(result["confidence"]["level"], "medium")

    def test_before_features_use_recent_form_without_future_result(self) -> None:
        result = predict_match_probability(
            "A",
            "B",
            1.5,
            1.5,
            stats_a={
                "career_win_rate_before": 0.60,
                "season_win_rate_before": 0.70,
                "last5_win_rate_before": 0.80,
                "last10_win_rate_before": 0.70,
                "h2h_win_rate_before": 0.65,
            },
            stats_b={
                "career_win_rate_before": 0.40,
                "season_win_rate_before": 0.30,
                "last5_win_rate_before": 0.20,
                "last10_win_rate_before": 0.30,
                "h2h_win_rate_before": 0.35,
            },
        )
        self.assertGreater(result["playerA"]["winProbability"], 0.5)
        metrics = {factor["metric"] for factor in result["keyFactors"]}
        self.assertTrue(metrics & {"시즌 승률", "최근 5경기", "최근 10경기"})


if __name__ == "__main__":
    unittest.main()
