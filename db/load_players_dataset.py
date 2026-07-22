# final_dataset_2026_start(선수 마스터·전적·Elo·시즌통계 CSV 11개)를
# PostgreSQL(RDS billiard)의 cuecast 스키마에 적재하는 로더.
# 각 테이블의 자연키로 upsert 하므로 몇 번을 재실행해도 중복이 안 생긴다(idempotent).
#
# 사용 예:
#   # 파싱·행수만 확인 (DB 미접속)
#   python db/load_players_dataset.py --dry-run
#
#   # 실제 적재 (db/db.env 의 DATABASE_URL 사용)
#   source db/db.env && python db/load_players_dataset.py
#
#   # 특정 테이블만
#   python db/load_players_dataset.py --only matches
#
# 필요 패키지: pip install -r db/requirements.txt  (psycopg2-binary)
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_SQL = os.path.join(HERE, "cuecast_schema.sql")
DEFAULT_ROOT = os.path.expanduser("~/Downloads/final_dataset_2026_start")

# ---------- 컬럼 타입 (변환/검증용) ----------
# 아래 집합에 없는 컬럼은 문자열로 두고 Postgres가 컬럼 타입(TEXT/DATE/TIMESTAMP)으로 캐스팅한다.
BOOL_COLS = {
    "active_2026_roster", "has_prior_career_stats", "appears_in_any_match",
    "image_available_for_service", "image_is_placeholder",
    "original_image_available", "is_walkover", "is_played", "model_eligible",
    "included_in_final_probability",
}
INT_COLS = {
    "season_code", "elo_matches_prior", "career_matches_prior",
    "career_wins_prior", "career_losses_prior", "last5_matches_prior",
    "last5_wins_prior", "last10_matches_prior", "last10_wins_prior",
    "prior_detail_matches", "season_matches", "season_wins", "season_losses",
    "recent_detail_matches_current_season", "standardization_roster_count",
    "recent_rank", "played_matches", "player_a_wins", "player_b_wins",
    "elo_matches_total", "matches", "wins", "losses", "detail_matches",
    "total_attempts", "successful_attempts", "break_attempts",
    "break_successes", "five_high_count", "chronological_index", "game_no",
    "player1_set_score", "player2_set_score", "player_side", "set_score",
    "pts", "inn", "five_high_shot_count", "break_shot_attempts",
    "break_shot_successes",
}
FLOAT_COLS = {
    "elo_start", "career_win_rate_raw", "career_win_rate_adjusted_t20",
    "career_confidence_component", "last5_win_rate_adjusted_t5",
    "last5_coverage_component", "last10_win_rate_adjusted_t10",
    "last10_coverage_component", "prior_innings", "prior_avg_raw",
    "prior_ts_pct_raw", "prior_brs_pct_raw", "prior_5hs_pct_raw",
    "prior_high_run_raw", "prior_detail_confidence_capped_150",
    "prior_avg_shrunk", "prior_ts_pct_shrunk", "prior_brs_pct_shrunk",
    "prior_5hs_pct_shrunk", "prior_high_run_shrunk",
    "season_win_rate_adjusted_t10", "season_innings", "season_avg",
    "season_ts_pct", "season_brs_pct", "season_5hs_pct", "season_high_run",
    "recent_detail_avg_current_season", "recent_detail_ts_pct_current_season",
    "recent_detail_brs_pct_current_season",
    "recent_detail_5hs_pct_current_season",
    "recent_detail_high_run_current_season", "final_avg_start",
    "final_ts_pct_start", "final_brs_pct_start", "final_5hs_pct_start",
    "final_high_run_start", "z_avg_start", "z_ts_start", "z_brs_start",
    "z_5hs_start", "z_hr_start", "performance_q_start", "prior_league_mean",
    "standardization_mean", "standardization_std_population", "win_rate_raw",
    "win_rate_adjusted_t10", "innings", "avg_weighted_sum", "avg", "ts_pct",
    "brs_pct", "five_hs_pct", "high_run", "elo_current", "success_rate_pct",
    "five_high_shot_pct", "break_shot_success_rate_pct", "mapping_score",
}

# ---------- 테이블 스펙: (테이블, [CSV 상대경로...], [PK 컬럼...]) ----------
TABLES = [
    ("player_master",
     ["01_players/player_master_2026.csv"],
     ["player_code"]),
    ("player_aliases",
     ["01_players/player_aliases.csv"],
     ["player_code", "alias"]),
    ("player_runtime_state",
     ["02_runtime/player_runtime_state_2026_start.csv"],
     ["player_code"]),
    ("league_metric_baselines",
     ["02_runtime/league_metric_baselines_2026_start.csv"],
     ["league", "metric"]),
    ("recent_match_events_seed",
     ["02_runtime/recent_match_events_seed_2026_start.csv"],
     ["league", "player_code", "recent_rank"]),
    ("head_to_head_reference",
     ["02_runtime/head_to_head_reference_2026_start.csv"],
     ["player_a_code", "player_b_code"]),
    ("season_player_state",
     ["03_current_season/season_player_state_2026.csv"],
     ["player_code"]),
    ("matches",
     ["04_rebuild/history_matches_pre_2026.csv",
      "03_current_season/matches_2026.csv"],
     ["match_uid"]),
    ("player_match_detail_stats",
     ["04_rebuild/history_player_detail_stats_pre_2026.csv",
      "03_current_season/player_match_detail_stats_2026.csv"],
     ["match_uid", "player_code"]),
]


def convert(col, raw):
    """CSV 셀 문자열 → 파이썬 값. 빈 문자열은 NULL. bool/int/float만 변환하고 나머지는 문자열."""
    if raw is None or raw == "":
        return None
    if col in BOOL_COLS:
        low = raw.strip().lower()
        if low in ("true", "t", "1"):
            return True
        if low in ("false", "f", "0"):
            return False
        raise ValueError(f"{col}: bool 아님 {raw!r}")
    if col in INT_COLS:
        return int(float(raw))          # "5.0" 같은 값도 허용
    if col in FLOAT_COLS:
        return float(raw)
    return raw                           # TEXT/DATE/TIMESTAMP 는 Postgres 가 캐스팅


def read_csv(path):
    """CSV → (컬럼명 리스트, 변환된 행 튜플 리스트). utf-8-sig 로 BOM 제거."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows = []
        for i, rec in enumerate(reader, start=2):  # 2 = 첫 데이터행 (헤더가 1행)
            try:
                rows.append(tuple(convert(c, rec.get(c)) for c in cols))
            except ValueError as e:
                raise ValueError(f"{os.path.basename(path)}:{i} {e}") from e
    return cols, rows


def upsert_sql(table, cols, pk):
    updatable = [c for c in cols if c not in pk]
    set_clause = ", ".join(f"{c}=EXCLUDED.{c}" for c in updatable)
    tail = (f"DO UPDATE SET {set_clause}, loaded_at=now()"
            if updatable else "DO NOTHING")
    return (
        f"INSERT INTO cuecast.{table} ({', '.join(cols)}) VALUES %s "
        f"ON CONFLICT ({', '.join(pk)}) {tail}"
    )


def load_table(cur, root, table, csv_rels, pk):
    from psycopg2.extras import execute_values
    total = 0
    header = None
    for rel in csv_rels:
        path = os.path.join(root, rel)
        cols, rows = read_csv(path)
        if header is None:
            header = cols
        elif cols != header:
            raise ValueError(f"{table}: 통합 CSV 헤더 불일치\n  {header}\n  {cols}")
        if rows:
            execute_values(cur, upsert_sql(table, cols, pk), rows, page_size=1000)
        total += len(rows)
        print(f"    {rel}: {len(rows)}행", flush=True)
    return total


def ensure_schema(conn):
    with open(SCHEMA_SQL) as f, conn.cursor() as cur:
        cur.execute(f.read())
    conn.commit()


def dry_run(root, only):
    print(f"[dry-run] 소스: {root}\n", flush=True)
    grand = 0
    for table, csv_rels, pk in TABLES:
        if only and table != only:
            continue
        n = 0
        header = None
        for rel in csv_rels:
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                sys.exit(f"없음: {path}")
            cols, rows = read_csv(path)
            if header is None:
                header = cols
            elif cols != header:
                sys.exit(f"{table}: 통합 CSV 헤더 불일치")
            n += len(rows)
            print(f"  {table} <- {rel}: {len(rows)}행, {len(cols)}컬럼", flush=True)
        print(f"  => {table} 총 {n}행 (PK {pk})\n", flush=True)
        grand += n
    print(f"[dry-run] 합계 {grand}행. 타입 변환 이상 없음.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="데이터셋 루트 폴더")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"),
                    help="postgres 접속 URL (기본: 환경변수 DATABASE_URL)")
    ap.add_argument("--only", metavar="TABLE", help="이 테이블만 적재")
    ap.add_argument("--dry-run", action="store_true",
                    help="DB 미접속, 파싱·행수·타입 검증만")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"데이터셋 폴더가 없습니다: {args.root}")

    if args.dry_run:
        dry_run(args.root, args.only)
        return

    if not args.database_url:
        sys.exit("DATABASE_URL 이 필요합니다 (환경변수 또는 --database-url). "
                 "실제 적재 없이 확인만 하려면 --dry-run.")

    import psycopg2
    conn = psycopg2.connect(args.database_url)
    try:
        ensure_schema(conn)
        print(f"소스: {args.root}", flush=True)
        grand = 0
        with conn.cursor() as cur:
            for table, csv_rels, pk in TABLES:
                if args.only and table != args.only:
                    continue
                print(f"  {table}:", flush=True)
                n = load_table(cur, args.root, table, csv_rels, pk)
                conn.commit()
                grand += n
                print(f"  => {table} {n}행 upsert", flush=True)
        print(f"\n적재 완료: 총 {grand}행", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
