from __future__ import annotations

import unittest

from cuecast_yolo.live_match_state import LiveMatchCoordinator


class _Provider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, str]] = []

    def fetch(
        self,
        player_a: str,
        player_b: str,
        *,
        set_number: int,
        format_key: str = "pba-default",
    ) -> dict[str, object]:
        self.calls.append((player_a, player_b, set_number, format_key))
        return {
            "playerA": {"name": player_a, "avgFinal": 1.7, "stats": {}},
            "playerB": {"name": player_b, "avgFinal": 1.4, "stats": {}},
            "format": {
                "key": format_key,
                "targetScore": 15,
                "setsToWin": 4,
            },
            "prematchProbabilityA": 0.5,
            "prematchSource": "dummy",
            "dataSource": "server_db",
        }


def _scoreboard(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "player1Score": 4,
        "player2Score": 3,
        "activeColor": "white",
        "row1Color": "white",
        "player1Name": "김영원",
        "player2Name": "김규준",
    }
    value.update(overrides)
    return value


class LiveMatchCoordinatorTest(unittest.TestCase):
    def test_waits_until_manual_names_and_shot_probability_exist(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        self.assertEqual(
            coordinator.update_scoreboard(
                _scoreboard(player1Name=None, player2Name=None)
            )["state"],
            "waiting",
        )
        self.assertEqual(coordinator.update_scoreboard(_scoreboard())["state"], "waiting")
        self.assertEqual(coordinator.update_shot(0.62, "white")["state"], "waiting")
        coordinator.set_player_names("김영원", "김재근")
        ready = coordinator.update_shot(0.62, "white")
        self.assertEqual(ready["state"], "ready")

    def test_scores_are_usable_without_set_or_inning(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.set_player_names("김영원", "김규준")

        waiting = coordinator.update_scoreboard(
            {
                "player1Score": 4,
                "player2Score": 3,
                "row1Color": "white",
            }
        )
        self.assertEqual(waiting["state"], "waiting")
        self.assertIn("포메이션", waiting["detail"])

        coordinator.update_scoreboard(
            {
                "activeColor": "white",
            }
        )
        ready = coordinator.update_shot(0.62, "white")

        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["result"]["inputs"]["scoreA"], 4)
        self.assertEqual(ready["result"]["inputs"]["scoreB"], 3)

    def test_uses_server_db_player_avg_in_result(self) -> None:
        provider = _Provider()
        coordinator = LiveMatchCoordinator(provider)
        coordinator.set_player_names("김영원", "김규준")
        coordinator.update_scoreboard(_scoreboard())
        status = coordinator.update_shot(0.62, "white")
        result = status["result"]
        self.assertEqual(result["playerA"]["avgFinal"], 1.7)
        self.assertEqual(result["playerB"]["avgFinal"], 1.4)
        self.assertEqual(result["dataSource"], "server_db")
        self.assertEqual(provider.calls[-1][:2], ("김영원", "김규준"))

    def test_score_change_invalidates_previous_layout_probability(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.set_player_names("김영원", "김규준")
        coordinator.update_scoreboard(_scoreboard())
        self.assertEqual(coordinator.update_shot(0.62, "white")["state"], "ready")
        status = coordinator.update_scoreboard(_scoreboard(player1Score=5))
        self.assertEqual(status["state"], "waiting")
        self.assertIn("포메이션", status["detail"])

    def test_live_probability_is_scoped_to_current_set(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.set_player_names("김영원", "김규준")
        coordinator.update_scoreboard(_scoreboard())
        ready = coordinator.update_shot(0.55, "white")
        self.assertEqual(ready["result"]["probabilityScope"], "current_set")
        self.assertEqual(ready["result"]["inputs"]["setsToWin"], 1)
        self.assertIn("setWinProbabilityA", ready["result"])

    def test_prematch_probability_is_available_before_shot_probability_is_known(
        self,
    ) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        status = coordinator.set_player_names("김영원", "김규준")
        self.assertEqual(status["state"], "waiting")
        self.assertIsNone(status["result"])
        self.assertEqual(status["prematch"]["prematchMatchProbabilityA"], 0.5)
        self.assertEqual(status["prematch"]["playerA"]["name"], "김영원")
        self.assertEqual(status["prematch"]["playerB"]["name"], "김규준")
        # 폴링에 쓰이는 status()도 (직접 반환값이 아니라) 같은 미리보기를 들고 있어야 한다.
        self.assertEqual(
            coordinator.status()["prematch"]["playerA"]["name"], "김영원"
        )
        self.assertEqual(coordinator._scoreboard, None)

    def test_manual_names_are_used_for_database_lookup(self) -> None:
        provider = _Provider()
        coordinator = LiveMatchCoordinator(provider)
        coordinator.update_scoreboard(_scoreboard())
        coordinator.set_player_names("김재근", "조재호")
        ready = coordinator.update_shot(0.55, "white")
        self.assertEqual(provider.calls[-1][:2], ("김재근", "조재호"))
        self.assertEqual(ready["result"]["playerNameSource"], "manual")

if __name__ == "__main__":
    unittest.main()
