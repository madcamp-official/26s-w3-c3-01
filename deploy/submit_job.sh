#!/usr/bin/env bash
# 로컬(노트북)에서 실행: 유튜브 링크를 큐에 넣는다. GPU 인스턴스가 꺼져 있으면 켠 뒤 등록.
# 등록만 하면 되고, 처리·정지는 인스턴스 쪽 worker_autostop.sh 가 알아서 한다.
#
# 사용법:
#   ./deploy/submit_job.sh "https://www.youtube.com/watch?v=XXXX" [작업이름]
#   ./deploy/submit_job.sh "https://youtu.be/AAAA" "https://youtu.be/BBBB"   # 여러 개도 가능
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config.sh"

if [ "$#" -lt 1 ]; then
  echo "사용법: $0 <youtube_url> [<youtube_url> ...]"
  exit 1
fi

# 1) 인스턴스 상태 확인, 멈춰있으면 켜고 SSH 열릴 때까지 대기
STATE=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --instance-ids "$GPU_INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

if [ "$STATE" != "running" ]; then
  echo "GPU 인스턴스가 '$STATE' 상태 → 시작합니다..."
  aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$GPU_INSTANCE_ID" >/dev/null
  aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$GPU_INSTANCE_ID"
  echo "인스턴스 running. SSH 접속 대기 중..."
  for _ in $(seq 1 40); do
    if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        "$SSH_HOST" true 2>/dev/null; then
      break
    fi
    sleep 5
  done
fi

# 2) 링크를 각각 큐 파일로 등록
for URL in "$@"; do
  NAME="job_$(date +%Y%m%d_%H%M%S)_$RANDOM"
  ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SSH_HOST" \
    "mkdir -p '$REPO_DIR/jobs/pending' && printf '%s\n' '$URL' > '$REPO_DIR/jobs/pending/$NAME.txt'"
  echo "큐 등록: $NAME  ←  $URL"
done

echo "완료. 인스턴스의 워커가 처리 후 유휴 상태가 되면 자동으로 스스로 정지합니다."
