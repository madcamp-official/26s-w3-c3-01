from __future__ import annotations

import unittest

from cuecast_yolo.prematch_live_inputs import PrematchLiveInputProvider
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
