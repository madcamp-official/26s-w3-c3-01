-- 3쿠션 턴 데이터 스키마 (PostgreSQL)
-- load_to_db.py 가 최초 실행 시 자동으로 이 DDL을 적용한다. 수동 적용도 가능:
--   psql "$DATABASE_URL" -f db/schema.sql

CREATE TABLE IF NOT EXISTS billiard_turns (
    video_id            TEXT    NOT NULL,   -- 유튜브 video id
    turn                INT     NOT NULL,   -- 영상 내 턴 순번(1부터)
    epoch               INT,                -- 연속 구간(클립) 번호. 값이 튀면 카메라 컷 경계
    shooter             TEXT,               -- 수구: white | yellow | red
    success             BOOLEAN,            -- 3쿠션 성공 여부 (판정 불가 시 NULL)
    success_method      TEXT,               -- trajectory | insufficient
    coverage            REAL,               -- 샷 구간 수구 관측 비율(0~1)
    cushions_before_2nd INT,                -- 2번째 목적구 접촉 전 쿠션 수
    bank_shot           BOOLEAN,            -- 뱅크샷(점수 +2) 여부 — 점수판 판정에서만 채워짐
    hits                JSONB,              -- 접촉 순서 예: ["red","white"]
    before_pos          JSONB   NOT NULL,   -- 샷 직전 좌표 {"white":[x,y],"yellow":[..],"red":[..]}
    after_pos           JSONB   NOT NULL,   -- 샷 이후 좌표 (동일 구조)
    after_source        TEXT,               -- settled(정지 확인) | last_seen(마지막 관측)
    frame_start         INT,
    frame_end           INT,
    time_start_s        REAL,
    time_end_s          REAL,
    loaded_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (video_id, turn)            -- 재적재 시 중복 없이 갱신(upsert)되는 자연키
);

-- 기존 테이블 마이그레이션 (CREATE IF NOT EXISTS 는 컬럼을 추가하지 않으므로)
ALTER TABLE billiard_turns ADD COLUMN IF NOT EXISTS bank_shot BOOLEAN;

CREATE INDEX IF NOT EXISTS idx_billiard_turns_video   ON billiard_turns (video_id);
CREATE INDEX IF NOT EXISTS idx_billiard_turns_shooter ON billiard_turns (shooter);
CREATE INDEX IF NOT EXISTS idx_billiard_turns_success ON billiard_turns (success);

-- 영상 단위 적재 이력 (몇 턴을 언제 넣었는지)
CREATE TABLE IF NOT EXISTS billiard_ingest_log (
    video_id    TEXT PRIMARY KEY,
    n_turns     INT,
    loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
