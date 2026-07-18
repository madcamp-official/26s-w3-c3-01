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


if __name__ == "__main__":
    unittest.main()
