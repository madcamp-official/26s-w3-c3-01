# S3(또는 로컬)의 turns.jsonl 을 읽어 PostgreSQL 에 적재하는 로더.
# (video_id, turn) 자연키로 upsert 하므로 몇 번을 재실행해도 중복이 안 생긴다.
#
# 사용 예:
#   # 특정 영상 하나 (S3)
#   DATABASE_URL=postgres://user:pw@host:5432/db S3_BUCKET=s3://my-bucket \
#     python db/load_to_db.py --video-id WV3tL6z3cqo
#
#   # S3 prefix 아래 모든 영상 (cron으로 주기 실행하기 좋음 — idempotent)
#   DATABASE_URL=... S3_BUCKET=s3://my-bucket python db/load_to_db.py --all
#
#   # S3 없이 로컬 results/ 폴더에서 (테스트용)
#   DATABASE_URL=... python db/load_to_db.py --all --local results
#
# 필요 패키지: pip install -r db/requirements.txt  (psycopg2-binary, boto3)
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_SQL = os.path.join(HERE, "schema.sql")

COLUMNS = ("video_id", "turn", "epoch", "shooter", "success", "success_method",
           "coverage", "cushions_before_2nd", "hits", "before_pos", "after_pos",
           "after_source", "frame_start", "frame_end", "time_start_s", "time_end_s",
           "bank_shot")

# execute_values 템플릿: hits/before_pos/after_pos(9,10,11번째)는 json 문자열 → jsonb 캐스팅
ROW_TEMPLATE = ("(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s)")

UPSERT_SQL = f"""
INSERT INTO billiard_turns ({", ".join(COLUMNS)}) VALUES %s
ON CONFLICT (video_id, turn) DO UPDATE SET
  epoch=EXCLUDED.epoch, shooter=EXCLUDED.shooter, success=EXCLUDED.success,
  success_method=EXCLUDED.success_method, coverage=EXCLUDED.coverage,
  cushions_before_2nd=EXCLUDED.cushions_before_2nd, hits=EXCLUDED.hits,
  before_pos=EXCLUDED.before_pos, after_pos=EXCLUDED.after_pos,
  after_source=EXCLUDED.after_source, frame_start=EXCLUDED.frame_start,
  frame_end=EXCLUDED.frame_end, time_start_s=EXCLUDED.time_start_s,
  time_end_s=EXCLUDED.time_end_s, bank_shot=EXCLUDED.bank_shot, loaded_at=now();
"""


def row_of(rec):
    """turns.jsonl 한 줄(dict) → DB 컬럼 순서에 맞는 튜플. json 필드는 문자열로 직렬화."""
    d = rec.get("success_detail") or {}
    return (
        rec["video_id"], rec["turn"], rec.get("epoch"), rec.get("shooter"),
        rec.get("success"), d.get("method"), d.get("coverage"),
        d.get("cushions_before_2nd"), json.dumps(d.get("hits") or []),
        json.dumps(rec["before"]), json.dumps(rec["after"]), rec.get("after_source"),
        rec.get("frame_start"), rec.get("frame_end"),
        rec.get("time_start_s"), rec.get("time_end_s"),
        d.get("bank_shot"),
    )


def parse_jsonl(text):
    """turns.jsonl 텍스트 → 레코드 dict 리스트 (빈 줄 무시)."""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------- 소스별 turns.jsonl 수집 ----------
def _bucket_name(s3_bucket):
    return s3_bucket[len("s3://"):] if s3_bucket.startswith("s3://") else s3_bucket


def iter_local(local_root, video_id):
    """로컬 results 폴더에서 (video_id, jsonl텍스트) 산출."""
    if video_id:
        paths = [os.path.join(local_root, video_id, "turns.jsonl")]
    else:
        paths = sorted(glob.glob(os.path.join(local_root, "*", "turns.jsonl")))
    for p in paths:
        if not os.path.exists(p):
            print(f"  건너뜀(없음): {p}", flush=True)
            continue
        vid = os.path.basename(os.path.dirname(p))
        with open(p) as f:
            yield vid, f.read()


def iter_s3(s3_bucket, prefix, video_id):
    """S3 에서 (video_id, jsonl텍스트) 산출."""
    import boto3
    s3 = boto3.client("s3")
    bucket = _bucket_name(s3_bucket)
    if video_id:
        keys = [f"{prefix}/{video_id}/turns.jsonl"]
    else:
        keys = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/turns.jsonl"):
                    keys.append(obj["Key"])
    for key in sorted(keys):
        vid = key.split("/")[-2]
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        except s3.exceptions.NoSuchKey:
            print(f"  건너뜀(없음): s3://{bucket}/{key}", flush=True)
            continue
        yield vid, body


# ---------- DB 적재 ----------
def load(conn, sources):
    from psycopg2.extras import execute_values
    total_videos = total_turns = 0
    with conn.cursor() as cur:
        for vid, text in sources:
            recs = parse_jsonl(text)
            if not recs:
                print(f"  {vid}: 턴 0개 — 건너뜀", flush=True)
                continue
            rows = [row_of(r) for r in recs]
            execute_values(cur, UPSERT_SQL, rows, template=ROW_TEMPLATE)
            # 재추출로 턴 수가 줄었을 때 남는 옛 행(유령 데이터) 제거 —
            # turns.jsonl 은 항상 영상 전체 추출본이므로 그보다 큰 turn 은 구버전 잔재다.
            cur.execute("DELETE FROM billiard_turns WHERE video_id=%s AND turn > %s",
                        (vid, len(recs)))
            cur.execute(
                "INSERT INTO billiard_ingest_log (video_id, n_turns, loaded_at) "
                "VALUES (%s, %s, now()) "
                "ON CONFLICT (video_id) DO UPDATE SET n_turns=EXCLUDED.n_turns, "
                "loaded_at=now()",
                (vid, len(recs)))
            conn.commit()
            print(f"  {vid}: {len(recs)}턴 upsert 완료", flush=True)
            total_videos += 1
            total_turns += len(recs)
    return total_videos, total_turns


def ensure_schema(conn):
    with open(SCHEMA_SQL) as f, conn.cursor() as cur:
        cur.execute(f.read())
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="이 영상 하나만 적재")
    g.add_argument("--all", action="store_true", help="prefix 아래 모든 영상 적재")
    ap.add_argument("--local", metavar="DIR",
                    help="S3 대신 로컬 results 폴더에서 읽기 (테스트용)")
    ap.add_argument("--bucket", default=os.environ.get("S3_BUCKET", ""),
                    help="S3 버킷 (기본: 환경변수 S3_BUCKET)")
    ap.add_argument("--prefix", default=os.environ.get("S3_PREFIX", "results").strip("/"))
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL"),
                    help="postgres 접속 URL (기본: 환경변수 DATABASE_URL)")
    args = ap.parse_args()

    if not args.database_url:
        sys.exit("DATABASE_URL 이 필요합니다 (환경변수 또는 --database-url).")

    video_id = None if args.all else args.video_id
    if args.local:
        sources = iter_local(args.local, video_id)
        print(f"소스: 로컬 {args.local}", flush=True)
    else:
        if not args.bucket:
            sys.exit("S3 모드에는 S3_BUCKET(또는 --bucket)이 필요합니다. 로컬은 --local 사용.")
        sources = iter_s3(args.bucket, args.prefix, video_id)
        print(f"소스: s3://{_bucket_name(args.bucket)}/{args.prefix}/", flush=True)

    import psycopg2
    conn = psycopg2.connect(args.database_url)
    try:
        ensure_schema(conn)
        n_vid, n_turn = load(conn, sources)
        print(f"\n적재 완료: 영상 {n_vid}개, 턴 {n_turn}행", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
