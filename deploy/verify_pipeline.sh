#!/usr/bin/env bash
# 로컬 맥에서 실행: 파이프라인 각 단계를 순서대로 점검한다.
#   bash deploy/verify_pipeline.sh
# 전제: ~/.ssh/config 에 Host 별칭 'billiard' 이 설정돼 있어야 한다(ssh billiard 로 서버 접속).
set -uo pipefail
H=billiard
BUCKET=billiard-turns-mcamp3w
REGION=ap-northeast-2

echo "════════ 1) 로컬 → EC2 SSH 접속 ════════"
ssh -o ConnectTimeout=10 "$H" 'echo "  OK: $(whoami)@$(hostname)"' \
  || { echo "  ❌ 접속 실패 — IP 바뀌었는지(~/.ssh/config HostName) / SG / 인스턴스 켜짐 확인"; exit 1; }

echo; echo "════════ 2) EC2 → S3 권한 (results/ 목록) ════════"
ssh "$H" "aws s3 ls s3://$BUCKET/results/ --region $REGION | sed 's/^/  /'"

echo; echo "════════ 3) 자동 적재 크론 + 최근 로그 ════════"
ssh "$H" 'crontab -l | grep load_to_db | sed "s/^/  cron: /"; echo "  --- loader.log 마지막 3줄 ---"; tail -3 ~/loader/loader.log | sed "s/^/  /"'

echo; echo "════════ 4) 서버 DB(RDS) 저장 현황 ════════"
ssh "$H" 'cd ~/loader && . ./loader.env && ./venv/bin/python - <<PY 2>/dev/null
import os, psycopg2
c = psycopg2.connect(os.environ["DATABASE_URL"]); cur = c.cursor()
cur.execute("select count(*) from billiard_turns"); print("  총 턴:", cur.fetchone()[0])
cur.execute("select video_id, count(*) from billiard_turns group by video_id order by video_id")
for v, n in cur.fetchall(): print(f"    {v}: {n}턴")
cur.execute("select max(loaded_at) from billiard_turns"); print("  마지막 적재:", cur.fetchone()[0])
c.close()
PY'

echo; echo "════════ 5) DB → S3 export 파일 존재 확인 ════════"
ssh "$H" "aws s3 ls s3://$BUCKET/exports/billiard_turns.jsonl --region $REGION | sed 's/^/  /'"

echo; echo "════════ 6) 서버 → 로컬 내려받기(예시) ════════"
echo "  scp $H:~/loader/billiard_turns_export.jsonl ~/26s-w3-c3-01/"

echo; echo "점검 끝."
