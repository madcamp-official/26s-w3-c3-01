#!/usr/bin/env bash
# EC2 로더에서 실행: 10분마다 S3 → RDS 적재하는 cron 등록 + 즉시 1회 동작 확인.
# 여러 번 실행해도 중복 등록되지 않는다(기존 load_to_db 줄을 지우고 새로 넣음).
#
# 사용법 (EC2 브라우저 터미널에서 한 줄):
#   cd ~/loader && aws s3 cp s3://billiard-turns-mcamp3w/code/setup_cron.sh . --region ap-northeast-2 && bash setup_cron.sh

# Amazon Linux 2023에는 cron이 기본 설치돼 있지 않다 → 없으면 설치하고 서비스 시작
if ! command -v crontab >/dev/null 2>&1; then
  echo "cron 미설치 → 설치합니다 (Amazon Linux 2023)"
  sudo dnf install -y cronie
  sudo systemctl enable --now crond
fi

LOADER_DIR="$HOME/loader"
CRON_LINE="*/10 * * * * cd $LOADER_DIR && . ./loader.env && ./venv/bin/python load_to_db.py --all >> $LOADER_DIR/loader.log 2>&1"

# 기존 load_to_db 관련 cron 줄 제거 후 새로 추가 (idempotent)
( crontab -l 2>/dev/null | grep -v 'load_to_db.py' ; echo "$CRON_LINE" ) | crontab -

echo "=== 등록된 crontab ==="
crontab -l

echo ""
echo "=== 즉시 1회 실행 (동작 확인) ==="
cd "$LOADER_DIR" && . ./loader.env && ./venv/bin/python load_to_db.py --all

echo ""
echo "완료: 앞으로 10분마다 자동으로 S3 → RDS 적재됩니다."
echo "로그 확인: tail -f $LOADER_DIR/loader.log"
