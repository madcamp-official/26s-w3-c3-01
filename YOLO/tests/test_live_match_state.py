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
        "set": 1,
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
    def test_waits_until_scoreboard_names_and_shot_probability_exist(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        self.assertEqual(
            coordinator.update_scoreboard(
                _scoreboard(player1Name=None, player2Name=None)
            )["state"],
            "waiting",
        )
        self.assertEqual(coordinator.update_scoreboard(_scoreboard())["state"], "waiting")
        ready = coordinator.update_shot(0.62, "white")
        self.assertEqual(ready["state"], "ready")

    def test_partial_scoreboard_fields_are_merged_without_waiting_for_inning(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())

        waiting = coordinator.update_scoreboard(
            {
                "player1Score": 4,
                "player2Score": 3,
                "row1Color": "white",
            }
        )
        self.assertEqual(waiting["state"], "waiting")
        self.assertIn("세트", waiting["detail"])

        coordinator.update_scoreboard(
            {
                "set": 1,
                "player1Name": "김영원",
                "player2Name": "김규준",
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
        coordinator.update_scoreboard(_scoreboard())
        status = coordinator.update_shot(0.62, "white")
        result = status["result"]
        self.assertEqual(result["playerA"]["avgFinal"], 1.7)
        self.assertEqual(result["playerB"]["avgFinal"], 1.4)
        self.assertEqual(result["dataSource"], "server_db")
        self.assertEqual(provider.calls[-1][:2], ("김영원", "김규준"))

    def test_score_change_invalidates_previous_layout_probability(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.update_scoreboard(_scoreboard())
        self.assertEqual(coordinator.update_shot(0.62, "white")["state"], "ready")
        status = coordinator.update_scoreboard(_scoreboard(player1Score=5))
        self.assertEqual(status["state"], "waiting")
        self.assertIn("포메이션", status["detail"])

    def test_set_transition_tracks_winner_seen_in_current_session(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.update_scoreboard(_scoreboard(player1Score=15, player2Score=11))
        coordinator.update_shot(0.6, "white")
        coordinator.update_scoreboard(
            _scoreboard(set=2, player1Score=0, player2Score=0, activeColor="yellow")
        )
        ready = coordinator.update_shot(0.55, "yellow")
        self.assertEqual(ready["result"]["setsWonA"], 1)
        self.assertEqual(ready["result"]["setsWonB"], 0)

    def test_mid_match_start_is_marked_provisional(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.update_scoreboard(_scoreboard(set=3))
        ready = coordinator.update_shot(0.55, "white")
        self.assertTrue(ready["result"]["setScoreProvisional"])
        self.assertEqual(ready["result"]["unknownCompletedSets"], 2)

    def test_skipped_set_numbers_are_marked_unknown(self) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        coordinator.update_scoreboard(_scoreboard(player1Score=15, player2Score=11))
        coordinator.update_scoreboard(
            _scoreboard(set=3, player1Score=0, player2Score=0)
        )
        ready = coordinator.update_shot(0.55, "white")
        self.assertEqual(ready["result"]["setsWonA"], 1)
        self.assertEqual(ready["result"]["unknownCompletedSets"], 1)

    def test_prematch_probability_is_available_before_shot_probability_is_known(
        self,
    ) -> None:
        coordinator = LiveMatchCoordinator(_Provider())
        status = coordinator.update_scoreboard(_scoreboard())
        self.assertEqual(status["state"], "waiting")
        self.assertIsNone(status["result"])
        self.assertEqual(status["prematch"]["prematchMatchProbabilityA"], 0.5)
        self.assertEqual(status["prematch"]["playerA"]["name"], "김영원")
        self.assertEqual(status["prematch"]["playerB"]["name"], "김규준")
        # 폴링에 쓰이는 status()도 (직접 반환값이 아니라) 같은 미리보기를 들고 있어야 한다.
        self.assertEqual(
            coordinator.status()["prematch"]["playerA"]["name"], "김영원"
        )

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
