# RDS(billiard_turns)를 JSONL로 내보내 S3에 업로드. EC2(로더)에서 실행.
# 맥은 RDS에 직접 못 붙으므로, EC2에서 뽑아 S3로 보낸 뒤 맥이 S3에서 받는다.
#
# 사용법 (EC2 브라우저 터미널):
#   cd ~/loader && aws s3 cp s3://billiard-turns-mcamp3w/code/export_db.py . --region ap-northeast-2
#   source loader.env && ./venv/bin/python export_db.py
#
# 결과: billiard_turns_export.jsonl (로컬) + s3://<버킷>/exports/billiard_turns.jsonl
import json
import os
import sys

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
S3_BUCKET = os.environ.get("S3_BUCKET", "").replace("s3://", "").strip()
OUT = "billiard_turns_export.jsonl"

if not DATABASE_URL:
    sys.exit("DATABASE_URL 이 필요합니다 (source loader.env 먼저).")

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("""
    SELECT video_id, turn, epoch, shooter, success, success_method, coverage,
           cushions_before_2nd, bank_shot, hits, before_pos, after_pos, after_source,
           frame_start, frame_end, time_start_s, time_end_s, loaded_at
    FROM billiard_turns
    ORDER BY video_id, turn
""")
cols = [d[0] for d in cur.description]

n = 0
with open(OUT, "w") as f:
    for row in cur:
        rec = dict(zip(cols, row))
        # jsonb(before_pos/after_pos/hits)는 psycopg2가 파이썬 객체로 파싱해 줌.
        # loaded_at(timestamp)만 문자열로.
        if rec.get("loaded_at") is not None:
            rec["loaded_at"] = rec["loaded_at"].isoformat()
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
conn.close()
print(f"{n}행 내보냄 → {OUT}")

# 영상별 건수 요약도 출력
import collections  # noqa: E402
per = collections.Counter()
for line in open(OUT):
    per[json.loads(line)["video_id"]] += 1
for vid, c in sorted(per.items()):
    print(f"  {vid}: {c}턴")

if S3_BUCKET:
    import boto3
    boto3.client("s3").upload_file(OUT, S3_BUCKET, "exports/billiard_turns.jsonl")
    print(f"\nS3 업로드 → s3://{S3_BUCKET}/exports/billiard_turns.jsonl")
    print("맥에서 받기: db/db.env source 후 → 아래 안내 참고")
else:
    print("S3_BUCKET 미설정 → 로컬 파일만 생성")
