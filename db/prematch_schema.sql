-- CueCast 경기 전 승률 예측용 2026 선수/스냅샷 스키마.
-- final_dataset_2026_start.zip은 db/import_prematch_dataset.py로 적재한다.

CREATE TABLE IF NOT EXISTS prematch_players (
    league                 TEXT NOT NULL CHECK (league IN ('PBA', 'LPBA')),
    player_code            TEXT NOT NULL,
    player_name            TEXT NOT NULL,
    player_name_short      TEXT NOT NULL,
    active_roster          BOOLEAN NOT NULL DEFAULT FALSE,
    image_is_placeholder   BOOLEAN NOT NULL DEFAULT FALSE,
    image_source_path      TEXT,
    image_mime_type        TEXT,
    image_bytes            BYTEA,
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (league, player_code)
);

CREATE TABLE IF NOT EXISTS prematch_player_features (
    snapshot_at                   TIMESTAMPTZ NOT NULL,
    league                        TEXT NOT NULL,
    season_code                   INT NOT NULL,
    player_code                   TEXT NOT NULL,
    elo                           DOUBLE PRECISION NOT NULL,
    career_matches                INT NOT NULL,
    career_wins                   INT NOT NULL,
    season_matches                INT NOT NULL,
    season_wins                   INT NOT NULL,
    last5_matches                 INT NOT NULL,
    last5_wins                    INT NOT NULL,
    last10_matches                INT NOT NULL,
    last10_wins                   INT NOT NULL,
    performance_score             DOUBLE PRECISION NOT NULL,
    performance_innings_total     DOUBLE PRECISION NOT NULL,
    metrics                       JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_cutoff_rule             TEXT,
    PRIMARY KEY (snapshot_at, league, player_code),
    FOREIGN KEY (league, player_code)
        REFERENCES prematch_players (league, player_code)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prematch_head_to_head (
    league                 TEXT NOT NULL,
    player_a_code          TEXT NOT NULL,
    player_b_code          TEXT NOT NULL,
    played_matches         INT NOT NULL,
    player_a_wins          INT NOT NULL,
    player_b_wins          INT NOT NULL,
    first_match_datetime   TIMESTAMPTZ,
    last_match_datetime    TIMESTAMPTZ,
    latest_winner_code     TEXT,
    latest_winner_name     TEXT,
    latest_match_uid       TEXT,
    PRIMARY KEY (league, player_a_code, player_b_code)
);

CREATE TABLE IF NOT EXISTS prematch_league_metric_baselines (
    snapshot_at                         TIMESTAMPTZ NOT NULL,
    league                              TEXT NOT NULL,
    season_code                         INT NOT NULL,
    metric                              TEXT NOT NULL,
    prior_league_mean                   DOUBLE PRECISION,
    standardization_mean                DOUBLE PRECISION,
    standardization_std_population      DOUBLE PRECISION,
    standardization_roster_count        INT,
    PRIMARY KEY (snapshot_at, league, metric)
);

CREATE INDEX IF NOT EXISTS idx_prematch_players_active
    ON prematch_players (league, active_roster, player_name);
CREATE INDEX IF NOT EXISTS idx_prematch_features_latest
    ON prematch_player_features (league, season_code, player_code, snapshot_at DESC);
CREATE INDEX IF NOT EXISTS idx_prematch_h2h_reverse
    ON prematch_head_to_head (league, player_b_code, player_a_code);
