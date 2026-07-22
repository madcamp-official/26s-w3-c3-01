from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from cuecast_yolo.prematch_live_inputs import (
    LockedScoreboardPlayerMatcher,
    PrematchLiveInputProvider,
    broadcast_name_similarity,
)
from cuecast_yolo.prematch_probability import PrematchDataError


class _PrematchService:
    source = "postgresql"

    def __init__(self, *, avg_b: float | None = 1.432) -> None:
        self.avg_b = avg_b
        self.list_calls: list[tuple[str, bool]] = []
        self.predict_calls: list[dict[str, object]] = []

    def list_players(self, league: str, active_only: bool = True):
        self.list_calls.append((league, active_only))
        if league == "PBA":
            return [
                {"code": "P001", "name": "김영원"},
                {"code": "P002", "name": "김규준"},
            ]
        return []

    def predict(self, payload):
        self.predict_calls.append(dict(payload))
        return {
            "playerA": {
                "code": "P001",
                "name": "김영원",
                "winProbability": 0.637,
                "metrics": {"AVG": 1.711},
            },
            "playerB": {
                "code": "P002",
                "name": "김규준",
                "winProbability": 0.363,
                "metrics": {"AVG": self.avg_b},
            },
            "modelVersion": "cuecast-prematch-linear-v1",
            "dataSource": "postgresql",
        }


class PrematchLiveInputProviderTest(unittest.TestCase):
    def test_db_matched_names_stay_fixed_until_explicit_reset(self) -> None:
        provider = MagicMock()
        provider.match_scoreboard_players.side_effect = [
            {
                "player1Name": "김영원",
                "player2Name": "김규준",
                "player1NameSimilarity": 0.9,
                "player2NameSimilarity": 0.9,
            },
            {
                "player1Name": "조재호",
                "player2Name": "조건휘",
                "player1NameSimilarity": 0.9,
                "player2NameSimilarity": 0.9,
            },
        ]
        matcher = LockedScoreboardPlayerMatcher(provider)

        first = matcher.match({"player1Name": "김영윈", "player2Name": "김규쥰"})
        self.assertTrue(matcher.locked)
        locked = matcher.match({"player1Name": "조재호", "player2Name": "조건휘"})
        matcher.reset()
        self.assertFalse(matcher.locked)
        refreshed = matcher.match({"player1Name": "조재호", "player2Name": "조건휘"})

        self.assertEqual(first["player1Name"], "김영원")
        self.assertEqual(locked["player1Name"], "김영원")
        self.assertEqual(refreshed["player1Name"], "조재호")
        self.assertEqual(provider.match_scoreboard_players.call_count, 2)

    def test_matches_misread_korean_scoreboard_names_to_distinct_db_players(self) -> None:
        service = _PrematchService()
        service.list_players = lambda league, active_only=True: (
            [
                {"code": "P001", "name": "김영원"},
                {"code": "P002", "name": "김규준"},
                {"code": "P003", "name": "김재근"},
                {"code": "P004", "name": "김동영"},
            ]
            if league == "PBA"
            else []
        )
        provider = PrematchLiveInputProvider(service)

        matched = provider.match_scoreboard_players(
            {"player1Name": "128강 김영윈", "player2Name": "김규쥰", "set": 1}
        )

        self.assertEqual(matched["player1Name"], "김영원")
        self.assertEqual(matched["player2Name"], "김규준")
        self.assertEqual(matched["player1OcrName"], "128강 김영윈")
        self.assertEqual(matched["player1League"], "PBA")

    def test_leaves_unrelated_ocr_name_unchanged(self) -> None:
        service = _PrematchService()
        provider = PrematchLiveInputProvider(service)

        matched = provider.match_scoreboard_players({"player1Name": "ABCXYZ"})

        self.assertEqual(matched["player1Name"], "ABCXYZ")
        self.assertNotIn("player1OcrName", matched)

    def test_broadcast_name_similarity_handles_title_prefix(self) -> None:
        self.assertGreaterEqual(
            broadcast_name_similarity("128강 #김영원", "김영원"), 0.9
        )

    def test_connects_remote_prematch_probability_and_db_avg(self) -> None:
        service = _PrematchService()
        provider = PrematchLiveInputProvider(service)

        result = provider.fetch("128강 #김영원", "#김규준", set_number=1)

        self.assertEqual(result["prematchProbabilityA"], 0.637)
        self.assertEqual(result["playerA"]["avgFinal"], 1.711)
        self.assertEqual(result["playerB"]["avgFinal"], 1.432)
        self.assertEqual(result["prematchSource"], "cuecast-prematch-linear-v1")
        self.assertEqual(service.predict_calls[0]["player_a_code"], "P001")
        self.assertEqual(service.predict_calls[0]["player_b_code"], "P002")

    def test_caches_player_lookup_and_prediction(self) -> None:
        service = _PrematchService()
        provider = PrematchLiveInputProvider(service)

        provider.fetch("김영원", "김규준", set_number=1)
        provider.fetch("김영원", "김규준", set_number=2)

        self.assertEqual(len(service.predict_calls), 1)
        self.assertEqual(service.list_calls.count(("PBA", False)), 1)

    def test_missing_avg_does_not_fall_back_to_dummy_data(self) -> None:
        provider = PrematchLiveInputProvider(_PrematchService(avg_b=None))

        with self.assertRaisesRegex(PrematchDataError, "AVG"):
            provider.fetch("김영원", "김규준", set_number=1)


if __name__ == "__main__":
    unittest.main()
