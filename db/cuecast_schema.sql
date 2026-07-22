-- CueCast 2026 선수 전적/런타임 데이터셋 스키마 (PostgreSQL)
-- 원본: ~/Downloads/final_dataset_2026_start 의 CSV 11개
-- load_players_dataset.py 가 최초 실행 시 이 DDL을 자동 적용한다. 수동 적용도 가능:
--   psql "$DATABASE_URL" -f db/cuecast_schema.sql
--
-- 영상 3쿠션 턴 데이터(public.billiard_turns)와 격리하기 위해 전용 스키마 cuecast 사용.

CREATE SCHEMA IF NOT EXISTS cuecast;

-- 1) 선수 마스터 -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS cuecast.player_master (
    league                       TEXT,
    player_code                  TEXT NOT NULL,
    player_name                  TEXT,
    player_name_short            TEXT,
    active_2026_roster           BOOLEAN,
    has_prior_career_stats       BOOLEAN,
    appears_in_any_match         BOOLEAN,
    image_available_for_service  BOOLEAN,
    image_is_placeholder         BOOLEAN,
    image_status                 TEXT,
    image_file                   TEXT,     -- 로컬 서비스 이미지 상대경로(바이너리는 저장 안 함)
    image_url                    TEXT,
    original_image_available     BOOLEAN,
    master_source                TEXT,
    loaded_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_code)
);
CREATE INDEX IF NOT EXISTS idx_cc_master_league ON cuecast.player_master (league);

-- 2) 선수 별칭 -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cuecast.player_aliases (
    league            TEXT,
    player_code       TEXT NOT NULL,
    alias             TEXT NOT NULL,
    normalized_alias  TEXT,
    alias_source      TEXT,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_code, alias)
);
CREATE INDEX IF NOT EXISTS idx_cc_alias_norm ON cuecast.player_aliases (normalized_alias);

-- 3) 런타임 스냅샷 (2026 개막 직전, 선수당 1행) -----------------------------
CREATE TABLE IF NOT EXISTS cuecast.player_runtime_state (
    league                                 TEXT,
    season_code                            INTEGER,
    player_code                            TEXT NOT NULL,
    player_name                            TEXT,
    player_name_short                      TEXT,
    active_2026_roster                     BOOLEAN,
    image_file                             TEXT,
    image_is_placeholder                   BOOLEAN,
    elo_start                              DOUBLE PRECISION,
    elo_matches_prior                      INTEGER,
    career_matches_prior                   INTEGER,
    career_wins_prior                      INTEGER,
    career_losses_prior                    INTEGER,
    career_win_rate_raw                    DOUBLE PRECISION,
    career_win_rate_adjusted_t20           DOUBLE PRECISION,
    career_confidence_component            DOUBLE PRECISION,
    last5_matches_prior                    INTEGER,
    last5_wins_prior                       INTEGER,
    last5_win_rate_adjusted_t5             DOUBLE PRECISION,
    last5_coverage_component               DOUBLE PRECISION,
    last10_matches_prior                   INTEGER,
    last10_wins_prior                      INTEGER,
    last10_win_rate_adjusted_t10           DOUBLE PRECISION,
    last10_coverage_component              DOUBLE PRECISION,
    prior_detail_matches                   INTEGER,
    prior_innings                          DOUBLE PRECISION,
    prior_avg_raw                          DOUBLE PRECISION,
    prior_ts_pct_raw                       DOUBLE PRECISION,
    prior_brs_pct_raw                      DOUBLE PRECISION,
    prior_5hs_pct_raw                      DOUBLE PRECISION,
    prior_high_run_raw                     DOUBLE PRECISION,
    prior_detail_confidence_capped_150     DOUBLE PRECISION,
    prior_avg_shrunk                       DOUBLE PRECISION,
    prior_ts_pct_shrunk                    DOUBLE PRECISION,
    prior_brs_pct_shrunk                   DOUBLE PRECISION,
    prior_5hs_pct_shrunk                   DOUBLE PRECISION,
    prior_high_run_shrunk                  DOUBLE PRECISION,
    season_matches                         INTEGER,
    season_wins                            INTEGER,
    season_losses                          INTEGER,
    season_win_rate_adjusted_t10           DOUBLE PRECISION,
    season_innings                         DOUBLE PRECISION,
    season_avg                             DOUBLE PRECISION,
    season_ts_pct                          DOUBLE PRECISION,
    season_brs_pct                         DOUBLE PRECISION,
    season_5hs_pct                         DOUBLE PRECISION,
    season_high_run                        DOUBLE PRECISION,
    recent_detail_matches_current_season   INTEGER,
    recent_detail_avg_current_season       DOUBLE PRECISION,
    recent_detail_ts_pct_current_season    DOUBLE PRECISION,
    recent_detail_brs_pct_current_season   DOUBLE PRECISION,
    recent_detail_5hs_pct_current_season   DOUBLE PRECISION,
    recent_detail_high_run_current_season  DOUBLE PRECISION,
    final_avg_start                        DOUBLE PRECISION,
    final_ts_pct_start                     DOUBLE PRECISION,
    final_brs_pct_start                    DOUBLE PRECISION,
    final_5hs_pct_start                    DOUBLE PRECISION,
    final_high_run_start                   DOUBLE PRECISION,
    z_avg_start                            DOUBLE PRECISION,
    z_ts_start                             DOUBLE PRECISION,
    z_brs_start                            DOUBLE PRECISION,
    z_5hs_start                            DOUBLE PRECISION,
    z_hr_start                             DOUBLE PRECISION,
    performance_q_start                    DOUBLE PRECISION,
    snapshot_cutoff_rule                   TEXT,
    loaded_at                              TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_code)
);
CREATE INDEX IF NOT EXISTS idx_cc_runtime_league ON cuecast.player_runtime_state (league);

-- 4) 리그 지표 베이스라인 --------------------------------------------------
CREATE TABLE IF NOT EXISTS cuecast.league_metric_baselines (
    league                          TEXT NOT NULL,
    season_code                     INTEGER,
    metric                          TEXT NOT NULL,
    prior_league_mean               DOUBLE PRECISION,
    standardization_mean            DOUBLE PRECISION,
    standardization_std_population  DOUBLE PRECISION,
    standardization_roster_count    INTEGER,
    prior_mean_method               TEXT,
    standardization_population      TEXT,
    loaded_at                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (league, metric)
);

-- 5) 최근 승률 rolling window 시드 -----------------------------------------
CREATE TABLE IF NOT EXISTS cuecast.recent_match_events_seed (
    league             TEXT NOT NULL,
    player_code        TEXT NOT NULL,
    player_name        TEXT,
    recent_rank        INTEGER NOT NULL,
    match_uid          TEXT,
    match_datetime     TIMESTAMP,
    season_code        INTEGER,
    tournament_seq     TEXT,
    tournament_title   TEXT,
    round_code         TEXT,
    round_label        TEXT,
    competition_scope  TEXT,
    opponent_code      TEXT,
    opponent_name      TEXT,
    result             TEXT,
    loaded_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (league, player_code, recent_rank)
);
CREATE INDEX IF NOT EXISTS idx_cc_recent_player ON cuecast.recent_match_events_seed (player_code);

-- 6) 상대전적 참고 (화면 표시용, 최종 확률 미반영) -------------------------
CREATE TABLE IF NOT EXISTS cuecast.head_to_head_reference (
    league                        TEXT,
    player_a_code                 TEXT NOT NULL,
    player_b_code                 TEXT NOT NULL,
    played_matches                INTEGER,
    player_a_wins                 INTEGER,
    player_b_wins                 INTEGER,
    first_match_datetime          TIMESTAMP,
    last_match_datetime           TIMESTAMP,
    latest_winner_code            TEXT,
    latest_match_uid              TEXT,
    player_a_name                 TEXT,
    player_b_name                 TEXT,
    latest_winner_name            TEXT,
    included_in_final_probability BOOLEAN,
    loaded_at                     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_a_code, player_b_code)
);
CREATE INDEX IF NOT EXISTS idx_cc_h2h_a ON cuecast.head_to_head_reference (player_a_code);
CREATE INDEX IF NOT EXISTS idx_cc_h2h_b ON cuecast.head_to_head_reference (player_b_code);

-- 7) 2026 시즌 누적 상태 (mutable) -----------------------------------------
CREATE TABLE IF NOT EXISTS cuecast.season_player_state (
    league                          TEXT,
    season_code                     INTEGER,
    player_code                     TEXT NOT NULL,
    player_name                     TEXT,
    active_2026_roster              BOOLEAN,
    elo_current                     DOUBLE PRECISION,
    elo_matches_total               INTEGER,
    matches                         INTEGER,
    wins                            INTEGER,
    losses                          INTEGER,
    win_rate_raw                    DOUBLE PRECISION,
    win_rate_adjusted_t10           DOUBLE PRECISION,
    detail_matches                  INTEGER,
    innings                         DOUBLE PRECISION,
    avg_weighted_sum                DOUBLE PRECISION,
    avg                             DOUBLE PRECISION,
    total_attempts                  INTEGER,
    successful_attempts             INTEGER,
    ts_pct                          DOUBLE PRECISION,
    break_attempts                  INTEGER,
    break_successes                 INTEGER,
    brs_pct                         DOUBLE PRECISION,
    five_high_count                 INTEGER,
    five_hs_pct                     DOUBLE PRECISION,
    high_run                        DOUBLE PRECISION,
    recent_detail_match_count_max5  INTEGER,
    last_updated_match_datetime     TIMESTAMP,
    loaded_at                       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_code)
);

-- 8) 경기 결과 (과거 pre_2026 + 현재 2026 통합) ----------------------------
CREATE TABLE IF NOT EXISTS cuecast.matches (
    match_uid           TEXT NOT NULL,
    chronological_index INTEGER,
    assignment_code     TEXT,
    league              TEXT,
    season_code         INTEGER,
    tournament_seq      TEXT,
    tournament_title    TEXT,
    round_code          TEXT,
    round_label         TEXT,
    competition_scope   TEXT,
    game_no             INTEGER,
    match_date          DATE,
    match_time          TEXT,
    match_datetime      TIMESTAMP,
    player1_code        TEXT,
    player1_name        TEXT,
    player2_code        TEXT,
    player2_name        TEXT,
    player1_set_score   INTEGER,
    player2_set_score   INTEGER,
    winner_code         TEXT,
    winner_name         TEXT,
    loser_code          TEXT,
    loser_name          TEXT,
    is_walkover         BOOLEAN,
    is_played           BOOLEAN,
    model_eligible      BOOLEAN,
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_uid)
);
CREATE INDEX IF NOT EXISTS idx_cc_matches_league_season ON cuecast.matches (league, season_code);
CREATE INDEX IF NOT EXISTS idx_cc_matches_p1 ON cuecast.matches (player1_code);
CREATE INDEX IF NOT EXISTS idx_cc_matches_p2 ON cuecast.matches (player2_code);

-- 9) 선수별 경기 세부기록 (과거 + 현재 통합) -------------------------------
CREATE TABLE IF NOT EXISTS cuecast.player_match_detail_stats (
    match_uid                    TEXT NOT NULL,
    assignment_code              TEXT,
    league                       TEXT,
    season_code                  INTEGER,
    tournament_seq               TEXT,
    round_code                   TEXT,
    match_datetime               TIMESTAMP,
    player_side                  INTEGER,
    player_code                  TEXT NOT NULL,
    player_name                  TEXT,
    opponent_code                TEXT,
    opponent_name                TEXT,
    result                       TEXT,
    set_score                    INTEGER,
    pts_by_set                   TEXT,     -- 세트별 득점 "10,3,3" (콤마 포함 문자열)
    pts                          INTEGER,
    inn                          INTEGER,
    avg                          DOUBLE PRECISION,
    high_run                     DOUBLE PRECISION,
    total_attempts               INTEGER,
    successful_attempts          INTEGER,
    success_rate_pct             DOUBLE PRECISION,
    five_high_shot_count         INTEGER,
    five_high_shot_pct           DOUBLE PRECISION,
    break_shot_attempts          INTEGER,
    break_shot_successes         INTEGER,
    break_shot_success_rate_pct  DOUBLE PRECISION,
    mapping_status               TEXT,
    identity_match_method        TEXT,
    mapping_score                DOUBLE PRECISION,
    loaded_at                    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (match_uid, player_code)
);
CREATE INDEX IF NOT EXISTS idx_cc_detail_player ON cuecast.player_match_detail_stats (player_code);
CREATE INDEX IF NOT EXISTS idx_cc_detail_league_season ON cuecast.player_match_detail_stats (league, season_code);
