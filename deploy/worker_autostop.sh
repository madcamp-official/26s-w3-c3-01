#!/usr/bin/env bash
# GPU 인스턴스에서 cron으로 주기 실행: 큐를 처리하고, 처리할 게 없으면 자기 자신을 정지한다.
# 인스턴스가 정지된 동안에는 cron이 안 도므로, 최초 기동은 로컬의 submit_job.sh 가 담당한다.
#
# crontab 등록 (인스턴스에서, 3분마다):
#   */3 * * * * /home/ubuntu/26s-w3-c3-01/deploy/worker_autostop.sh >> /home/ubuntu/26s-w3-c3-01/logs/worker_autostop.log 2>&1
#
# 필요 권한: 이 인스턴스에 ec2:StopInstances 를 허용하는 IAM 역할이 붙어 있어야 한다.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"
cd "$REPO_DIR"
mkdir -p logs jobs/pending jobs/running
IDLE_FILE="$REPO_DIR/jobs/.idle_count"

echo "[$(date '+%F %T')] worker_autostop 시작"

# 1) 대기 중인 작업 처리. process_queue.py 자체의 mkdir 락이 중복 실행을 막으므로
#    cron이 겹쳐 떠도 실제 처리는 하나만 돈다(이미 처리 중이면 즉시 반환).
"$REPO_DIR/venv/bin/python" src/process_queue.py

# 2) 유휴 판정: pending 과 running 이 모두 비어야 '할 일 없음'
PENDING=$(find jobs/pending -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')
RUNNING=$(find jobs/running -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')

if [ "$PENDING" -eq 0 ] && [ "$RUNNING" -eq 0 ]; then
  N=$(( $(cat "$IDLE_FILE" 2>/dev/null || echo 0) + 1 ))
  echo "$N" > "$IDLE_FILE"
  echo "[$(date '+%F %T')] 유휴 사이클 $N/$IDLE_CYCLES_BEFORE_STOP"
  if [ "$N" -ge "$IDLE_CYCLES_BEFORE_STOP" ]; then
    echo "[$(date '+%F %T')] 처리할 작업 없음 → 인스턴스 정지"
    rm -f "$IDLE_FILE"
    # IMDSv2로 자기 인스턴스 ID를 조회해서 stop (config의 ID와 무관하게 '나'를 정지)
    TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
    IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$IID"
  fi
else
  # 할 일이 있으면 유휴 카운트 리셋
  rm -f "$IDLE_FILE"
  echo "[$(date '+%F %T')] 진행 중 (pending=$PENDING, running=$RUNNING) — 정지 안 함"
fi
