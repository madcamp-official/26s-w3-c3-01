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

    def test_player_name_editor_searches_postgres_player_list(self) -> None:
        self.assertNotIn("<datalist", self.html)
        self.assertIn('id="player1-name-results"', self.html)
        self.assertIn('id="player2-name-results"', self.html)
        self.assertIn('id="player-name-db-status"', self.html)
        self.assertIn("['PBA','LPBA']", self.html)
        self.assertIn("active_only=false", self.html)
        self.assertIn("loadNameEditorPlayers()", self.html)
        self.assertIn("button.textContent=player.name", self.html)
        self.assertNotIn("option.label=", self.html)
        self.assertIn("player1NameSimilarity", self.html)
        self.assertIn("db_match", self.html)

    def test_stats_player_picker_uses_the_same_search_dropdown(self) -> None:
        self.assertIn('id="player-a-results"', self.html)
        self.assertIn('id="player-b-results"', self.html)
        self.assertIn(
            "bindPlayerSearch('player-a','player-a-results'", self.html
        )
        self.assertIn("dataset.playerCode", self.html)

    def test_cuecast_logo_is_used_in_the_site_header(self) -> None:
        self.assertTrue(LOGO.is_file())
        self.assertTrue(FAVICON.is_file())
        self.assertIn('class="brand"', self.html)
        self.assertIn('src="/assets/logo.png"', self.html)
        self.assertNotIn("<span>AI 3쿠션 샷 분석</span>", self.html)
        self.assertIn('rel="icon" type="image/png" href="/assets/home.png"', self.html)

    def test_manual_names_are_kept_only_for_the_current_session(self) -> None:
        self.assertNotIn("cuecast-player-names:", self.html)
        self.assertNotIn("loadManualPlayerNames", self.html)
        self.assertNotIn("saveManualPlayerNames", self.html)
        self.assertIn("scoreboard.player1Name", self.html)
        self.assertIn("manualPlayerNames||lastDetectedPlayerNames", self.html)
        self.assertIn("/api/v1/live-match/players", self.html)

    def test_youtube_title_is_not_used_as_player_name_fallback(self) -> None:
        self.assertNotIn("playerNamesFromTitle", self.html)
        self.assertIn("이름 인식 중", self.html)

    def test_scoreboard_hides_set_and_inning(self) -> None:
        self.assertNotIn('id="scoreboard-set"', self.html)
        self.assertNotIn('id="scoreboard-inning"', self.html)
        self.assertNotIn("String(scoreboard.set)", self.html)

    def test_only_one_player_run_is_displayed(self) -> None:
        self.assertIn("runFields.forEach", self.html)
        self.assertIn("?'--':String(scoreboard[field])", self.html)
        self.assertIn("player1Run", self.html)
        self.assertIn("player2Run", self.html)

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

    def test_confirmed_shots_are_saved_only_after_score_or_turn_changes(self) -> None:
        self.assertIn('id="shot-history-list"', self.html)
        self.assertIn('id="clear-shot-history"', self.html)
        self.assertIn("cuecast-shot-history:${id}", self.html)
        self.assertIn("stageConfirmedShot(d)", self.html)
        self.assertIn("observeShotCompletion(scoreboard)", self.html)
        self.assertIn("scoreChanged||turnChanged", self.html)
        self.assertIn("recordConfirmedShot(pendingShotRecord", self.html)
        self.assertNotIn("recordConfirmedShot(d)", self.html)
        self.assertIn("data.confirmedBefore", self.html)

    def test_shot_panel_uses_similar_shot_count_instead_of_confidence(self) -> None:
        self.assertIn("<span>유사 샷 개수</span>", self.html)
        self.assertIn('id="neighbor-count"', self.html)
        self.assertIn("neighborRawSamples", self.html)
        self.assertNotIn('<span>분석 신뢰도</span><strong id="confidence"', self.html)

    def test_developer_details_include_hybrid_model_breakdown(self) -> None:
        self.assertNotIn("<h2>분석 요약</h2>", self.html)
        for element_id in (
            "developer-confidence",
            "developer-data-status",
            "records",
            "model",
            "component-model-probability",
            "component-model-weight",
            "component-neighbor-probability",
            "component-neighbor-weight",
            "component-grid-probability",
            "component-grid-weight",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("components.weights", self.html)
        self.assertIn("components.modelProbability", self.html)
        self.assertIn("components.neighborProbability", self.html)
        self.assertIn("components.gridProbability", self.html)

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
