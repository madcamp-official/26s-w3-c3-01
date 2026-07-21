#!/bin/bash
# 로컬에서 CueCast 를 띄운다: 유튜브 실시간 분석은 (봇 차단 없는) 이 Mac에서 하고,
# 성공확률은 서버 RDS 데이터로 학습된 하이브리드 엔진이 계산한다.
#
# 흐름: 서버 RDS export(billiard_turns_export.jsonl) 최신본을 scp 로 받아 --shots 로 로드
#       → http://127.0.0.1:8765 로 서빙.
# 사용:  ./run_cuecast_local.sh        (Ctrl+C 로 종료)

set -e
cd "$(dirname "$0")"

EXPORT=./billiard_turns_export.jsonl
PY=venv/bin/python

# --- opencv 버전 가드: 이 코드(scoreboard_reader)는 opencv<5 필요 (cv2.ml.KNearest_create) ---
if ! "$PY" -c "import cv2,sys; sys.exit(0 if int(cv2.__version__.split('.')[0])<5 else 1)" 2>/dev/null; then
  CUR=$("$PY" -c "import cv2;print(cv2.__version__)" 2>/dev/null || echo "?")
  echo "! opencv $CUR 감지 — 이 코드는 opencv<5 가 필요합니다. 아래로 정렬 후 다시 실행:"
  echo "    $PY -m pip install 'opencv-python>=4.10,<5'"
  exit 1
fi

# --- 1) 서버 최신 export(=DB 데이터) 받기 (실패해도 기존 로컬 파일로 진행) ---
echo "[1/2] 서버 최신 export 받는 중 (billiard:~/app/billiard_turns_export.jsonl) ..."
if scp -o ConnectTimeout=15 billiard:/home/ec2-user/app/billiard_turns_export.jsonl "$EXPORT" 2>/dev/null; then
  echo "  -> 최신본 수신: $(wc -l < "$EXPORT" | tr -d ' ') 레코드"
else
  echo "  ! 서버 접속 실패 — 기존 로컬 export 로 진행 ($( (wc -l < "$EXPORT" 2>/dev/null || echo 0) | tr -d ' ') 레코드)"
fi

# --- 2) CueCast 기동 (CatBoost 학습에 수십 초) ---
echo "[2/2] CueCast 기동 중 -> http://127.0.0.1:8765   (Ctrl+C 로 종료)"
exec "$PY" YOLO/local_probability_server.py --shots "$EXPORT" --host 127.0.0.1 --port 8765
