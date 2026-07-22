from __future__ import annotations

from threading import Lock
from typing import Protocol

from .live_match_probability import predict_live_match_probability


class LiveMatchInputProvider(Protocol):
    def fetch(
        self,
        player_a: str,
        player_b: str,
        *,
        set_number: int,
        format_key: str = "pba-default",
    ) -> dict[str, object]: ...


class LiveMatchCoordinator:
    """Combine scoreboard state, DB player inputs, and current layout odds."""

    def __init__(
        self,
        provider: LiveMatchInputProvider,
        *,
        format_key: str = "pba-default",
    ) -> None:
        self.provider = provider
        self.format_key = format_key
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._scoreboard: dict[str, object] | None = None
            self._shot_probability: float | None = None
            self._shot_color: str | None = None
            self._sets_won_a = 0
            self._sets_won_b = 0
            self._unknown_completed_sets = 0
            self._set_starting_player: str | None = None
            self._manual_names: tuple[str, str] | None = None
            self._previous_probability_a: float | None = None
            self._status: dict[str, object] = {
                "state": "waiting",
                "detail": "점수판과 현재 포메이션을 기다리는 중",
                "result": None,
            }

    @staticmethod
    def _player_for_color(scoreboard: dict[str, object], color: object) -> str | None:
        if color not in ("white", "yellow"):
            return None
        row1_color = scoreboard.get("row1Color")
        if row1_color not in ("white", "yellow"):
            return None
        return "a" if color == row1_color else "b"

    def update_scoreboard(self, scoreboard: dict[str, object]) -> dict[str, object]:
        with self._lock:
            previous = self._scoreboard
            current_set = int(scoreboard.get("set", 1))
            if previous is None and current_set > 1:
                self._unknown_completed_sets = current_set - 1
            if previous is not None:
                previous_set = int(previous.get("set", current_set))
                if current_set < previous_set:
                    self._sets_won_a = self._sets_won_b = 0
                    self._unknown_completed_sets = max(0, current_set - 1)
                    self._set_starting_player = None
                    self._previous_probability_a = None
                elif current_set > previous_set:
                    skipped_sets = max(0, current_set - previous_set - 1)
                    score_a = int(previous.get("player1Score", 0))
                    score_b = int(previous.get("player2Score", 0))
                    if score_a > score_b:
                        self._sets_won_a += 1
                    elif score_b > score_a:
                        self._sets_won_b += 1
                    else:
                        skipped_sets += 1
                    self._unknown_completed_sets += skipped_sets
                    self._set_starting_player = None
                    self._shot_probability = None
                else:
                    score_changed = any(
                        int(previous.get(key, -1)) != int(scoreboard.get(key, -1))
                        for key in ("player1Score", "player2Score")
                    )
                    player_changed = previous.get("activeColor") != scoreboard.get(
                        "activeColor"
                    )
                    if score_changed or player_changed:
                        self._shot_probability = None
            self._scoreboard = dict(scoreboard)
            current_player = self._player_for_color(
                self._scoreboard, self._scoreboard.get("activeColor")
            )
            if self._set_starting_player is None and current_player is not None:
                self._set_starting_player = current_player
            return self._calculate_locked()

    def update_shot(
        self, probability: float, shooter_color: str
    ) -> dict[str, object]:
        with self._lock:
            self._shot_probability = min(1.0, max(0.0, float(probability)))
            self._shot_color = shooter_color
            return self._calculate_locked()

    def set_player_names(
        self, player_a: str | None, player_b: str | None
    ) -> dict[str, object]:
        with self._lock:
            if player_a is None and player_b is None:
                self._manual_names = None
            elif not player_a or not player_a.strip() or not player_b or not player_b.strip():
                raise ValueError("두 선수 이름이 모두 필요합니다")
            else:
                self._manual_names = (player_a.strip(), player_b.strip())
            return self._calculate_locked()

    def status(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def _waiting(self, detail: str) -> dict[str, object]:
        self._status = {"state": "waiting", "detail": detail, "result": None}
        return dict(self._status)

    def _calculate_locked(self) -> dict[str, object]:
        scoreboard = self._scoreboard
        if scoreboard is None:
            return self._waiting("점수판 인식 대기 중")
        names = self._manual_names or (
            scoreboard.get("player1Name"),
            scoreboard.get("player2Name"),
        )
        if not all(isinstance(name, str) and name.strip() for name in names):
            return self._waiting("선수 이름 OCR 대기 중")
        if self._shot_probability is None:
            return self._waiting("현재 포메이션 성공률 대기 중")
        current_player = self._player_for_color(scoreboard, self._shot_color)
        if current_player is None:
            current_player = self._player_for_color(
                scoreboard, scoreboard.get("activeColor")
            )
        if current_player is None:
            return self._waiting("현재 공격자 판독 대기 중")
        starting_player = self._set_starting_player or current_player
        set_number = int(scoreboard.get("set", 1))
        try:
            db_inputs = self.provider.fetch(
                str(names[0]),
                str(names[1]),
                set_number=set_number,
                format_key=self.format_key,
            )
            player_a = db_inputs["playerA"]
            player_b = db_inputs["playerB"]
            match_format = db_inputs["format"]
            if not all(
                isinstance(value, dict)
                for value in (player_a, player_b, match_format)
            ):
                raise RuntimeError("DB 입력 형식이 올바르지 않습니다")
            result = predict_live_match_probability(
                prematch_probability_a=float(
                    db_inputs.get("prematchProbabilityA", 0.5)
                ),
                final_avg_a=float(player_a["avgFinal"]),
                final_avg_b=float(player_b["avgFinal"]),
                score_a=int(scoreboard.get("player1Score", 0)),
                score_b=int(scoreboard.get("player2Score", 0)),
                target_score=int(match_format["targetScore"]),
                starting_player=starting_player,
                current_player=current_player,
                current_shot_probability=self._shot_probability,
                sets_won_a=self._sets_won_a,
                sets_won_b=self._sets_won_b,
                sets_to_win=int(match_format["setsToWin"]),
                previous_probability_a=self._previous_probability_a,
            )
        except Exception as error:
            self._status = {
                "state": "error",
                "detail": str(error),
                "result": None,
            }
            return dict(self._status)
        self._previous_probability_a = float(result["matchWinProbabilityA"])
        result.update(
            {
                "playerA": player_a,
                "playerB": player_b,
                "format": match_format,
                "setNumber": set_number,
                "setsWonA": self._sets_won_a,
                "setsWonB": self._sets_won_b,
                "unknownCompletedSets": self._unknown_completed_sets,
                "setScoreProvisional": self._unknown_completed_sets > 0,
                "prematchSource": db_inputs.get("prematchSource", "dummy"),
                "dataSource": db_inputs.get("dataSource", "server_db"),
                "playerNameSource": "manual" if self._manual_names else "scoreboard_ocr",
            }
        )
        self._status = {"state": "ready", "detail": "계산 완료", "result": result}
        return dict(self._status)
