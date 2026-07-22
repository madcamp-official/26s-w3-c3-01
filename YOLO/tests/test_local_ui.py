from __future__ import annotations

from pathlib import Path
import unittest


UI = Path(__file__).resolve().parents[1] / "ui" / "index.html"
LOGO = UI.parent / "assets" / "logo.png"
FAVICON = UI.parent / "assets" / "home.png"


class LocalUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = UI.read_text(encoding="utf-8")

    def test_player_name_editor_controls_are_present(self) -> None:
        for element_id in (
            "edit-player-names",
            "player-name-dialog",
            "player1-name-input",
            "player2-name-input",
            "reset-player-names",
        ):
            self.assertIn(f'id="{element_id}"', self.html)

    def test_cuecast_logo_is_used_in_the_site_header(self) -> None:
        self.assertTrue(LOGO.is_file())
        self.assertTrue(FAVICON.is_file())
        self.assertIn('class="brand"', self.html)
        self.assertIn('src="/assets/logo.png"', self.html)
        self.assertIn('rel="icon" type="image/png" href="/assets/home.png"', self.html)

    def test_manual_names_are_saved_per_youtube_video(self) -> None:
        self.assertIn("cuecast-player-names:${id}", self.html)
        self.assertIn("scoreboard.player1Name", self.html)
        self.assertIn("manualPlayerNames||lastDetectedPlayerNames", self.html)
        self.assertIn("/api/v1/live-match/players", self.html)

    def test_youtube_title_is_not_used_as_player_name_fallback(self) -> None:
        self.assertNotIn("playerNamesFromTitle", self.html)
        self.assertIn("이름 인식 중", self.html)

    def test_scoreboard_keeps_set_and_hides_inning(self) -> None:
        self.assertIn('id="scoreboard-set"', self.html)
        self.assertNotIn('id="scoreboard-inning"', self.html)

    def test_scoreboard_can_be_reset_for_fresh_ocr(self) -> None:
        self.assertIn('id="refresh-scoreboard"', self.html)
        self.assertIn('/api/v1/youtube/live/scoreboard/reset', self.html)
        self.assertIn("clearScoreboardDisplay()", self.html)

    def test_cue_ball_can_be_selected_manually(self) -> None:
        self.assertIn('<button class="segment active" id="white"', self.html)
        self.assertIn('<button class="segment" id="yellow"', self.html)
        self.assertIn('/api/v1/youtube/live/shooter', self.html)
        self.assertIn("selectShooter('white')", self.html)
        self.assertIn("selectShooter('yellow')", self.html)
        self.assertIn("acceptDetectedShooter(s.shooterConfirmed?s.shooter:null)", self.html)

    def test_confirmed_shots_are_persisted_per_video(self) -> None:
        self.assertIn('id="shot-history-list"', self.html)
        self.assertIn('id="clear-shot-history"', self.html)
        self.assertIn("cuecast-shot-history:${id}", self.html)
        self.assertIn("recordConfirmedShot(d)", self.html)
        self.assertIn("data.confirmedBefore", self.html)

    def test_live_match_probability_uses_automatic_server_result(self) -> None:
        for element_id in (
            "live-match-probability-a",
            "live-match-probability-b",
            "live-set-probability",
            "live-match-change",
            "live-shot-conditions",
            "live-match-source",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("/api/v1/live-match-probability/latest", self.html)
        self.assertIn("경기 전 A ${prematchA.toFixed(1)}%", self.html)
        self.assertIn("DB AVG ${avgA.toFixed(3)}", self.html)


if __name__ == "__main__":
    unittest.main()
