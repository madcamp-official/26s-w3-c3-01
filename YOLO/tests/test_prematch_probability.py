from __future__ import annotations

import unittest

from cuecast_yolo.prematch_probability import (
    PlayerFeatureSnapshot,
    PrematchDataError,
    adjusted_rate,
    pair_probability,
    predict_from_snapshots,
)


def snapshot(
    code: str,
    *,
    elo: float,
    career_wins: int,
    performance_score: float,
) -> PlayerFeatureSnapshot:
    return PlayerFeatureSnapshot(
        league="PBA",
        season_code=2026,
        player_code=code,
        player_name=code,
        player_name_short=code,
        active_roster=True,
        image_is_placeholder=False,
        elo=elo,
        career_matches=40,
        career_wins=career_wins,
        season_matches=0,
        season_wins=0,
        last5_matches=5,
        last5_wins=3 if code == "A" else 2,
        last10_matches=10,
        last10_wins=6 if code == "A" else 4,
        performance_score=performance_score,
        performance_innings_total=200,
        metrics={"AVG": 1.5, "TS": 55.0},
    )


class PrematchProbabilityTest(unittest.TestCase):
    def test_adjusted_rate_uses_half_win_prior(self) -> None:
        self.assertEqual(adjusted_rate(0, 0, 20), 0.5)
        self.assertEqual(adjusted_rate(3, 5, 10), 0.55)
        self.assertEqual(adjusted_rate(8, 10, 10), 0.8)

    def test_pair_probability_is_symmetric(self) -> None:
        probability = pair_probability(0.6, 0.4)
        reverse = pair_probability(0.4, 0.6)
        self.assertAlmostEqual(probability + reverse, 1.0)

    def test_snapshot_prediction_is_complementary_and_swap_symmetric(self) -> None:
        player_a = snapshot("A", elo=1570, career_wins=25, performance_score=0.35)
        player_b = snapshot("B", elo=1490, career_wins=18, performance_score=-0.10)
        result = predict_from_snapshots(player_a, player_b)
        swapped = predict_from_snapshots(player_b, player_a)
        self.assertAlmostEqual(
            result["playerA"]["winProbability"]
            + result["playerB"]["winProbability"],
            1.0,
        )
        self.assertAlmostEqual(
            result["playerA"]["winProbability"],
            swapped["playerB"]["winProbability"],
        )
        self.assertGreater(result["playerA"]["winProbability"], 0.5)

    def test_head_to_head_is_reference_only(self) -> None:
        player_a = snapshot("A", elo=1500, career_wins=20, performance_score=0.0)
        player_b = snapshot("B", elo=1500, career_wins=20, performance_score=0.0)
        without_h2h = predict_from_snapshots(player_a, player_b)
        with_h2h = predict_from_snapshots(
            player_a,
            player_b,
            head_to_head={"matches": 20, "winsA": 20, "winsB": 0},
        )
        self.assertEqual(
            without_h2h["playerA"]["winProbability"],
            with_h2h["playerA"]["winProbability"],
        )
        self.assertFalse(with_h2h["headToHeadIncludedInProbability"])

    def test_same_player_is_rejected(self) -> None:
        player = snapshot("A", elo=1500, career_wins=20, performance_score=0.0)
        with self.assertRaises(PrematchDataError):
            predict_from_snapshots(player, player)


if __name__ == "__main__":
    unittest.main()
