#!/bin/bash
# 로컬에서 CueCast 를 띄운다: 유튜브 실시간 분석은 (봇 차단 없는) 이 Mac에서 하고,
# 성공확률은 서버 RDS 데이터로 학습된 하이브리드 엔진이 계산한다.
#
# 흐름: 서버 RDS export(billiard_turns_export.jsonl) 최신본을 scp 로 받아 --shots 로 로드
#       → 경기 전 승률/선수 검색용 RDS는 EC2(billiard) 경유 SSH 터널로 접속
#       → http://127.0.0.1:8765 로 서빙.
# 사용:  ./run_cuecast_local.sh        (Ctrl+C 로 종료)
#
# 사전 준비 (DB 접속, 선택):
#   1) ~/.ssh/config 에 Host billiard 별칭 등록 (ec2-user@<EC2 IP>, IdentityFile ~/.ssh/billiard_ec2)
#   2) YOLO/.env (git에 포함 안 됨, 팀원에게 비밀번호 받아 각자 로컬에 생성) 에
#        DATABASE_URL=postgres://<user>:<password>@localhost:15432/<dbname>
#      캠퍼스 네트워크에서는 RDS:5432 직접 접속이 막혀 있어 반드시 localhost 터널을 거쳐야 한다.
#   위 두 가지가 없어도 스크립트는 DB 없이(경기 전 승률/선수 검색만 비활성) 계속 진행한다.

set -e
cd "$(dirname "$0")"

EXPORT=./billiard_turns_export.jsonl
PY=venv/bin/python

SSH_ALIAS=${CUECAST_SSH_ALIAS:-billiard}
RDS_HOST=${CUECAST_RDS_HOST:-billiard-db.cx0e6meogl9d.ap-northeast-2.rds.amazonaws.com}
TUNNEL_PORT=${CUECAST_TUNNEL_PORT:-15432}

# --- opencv 버전 가드: 이 코드(scoreboard_reader)는 opencv<5 필요 (cv2.ml.KNearest_create) ---
if ! "$PY" -c "import cv2,sys; sys.exit(0 if int(cv2.__version__.split('.')[0])<5 else 1)" 2>/dev/null; then
  CUR=$("$PY" -c "import cv2;print(cv2.__version__)" 2>/dev/null || echo "?")
  echo "! opencv $CUR 감지 — 이 코드는 opencv<5 가 필요합니다. 아래로 정렬 후 다시 실행:"
  echo "    $PY -m pip install 'opencv-python>=4.10,<5'"
  exit 1
fi

# --- 1) 서버 최신 export(=DB 데이터) 받기 (실패해도 기존 로컬 파일로 진행) ---
echo "[1/3] 서버 최신 export 받는 중 (${SSH_ALIAS}:~/app/billiard_turns_export.jsonl) ..."
if scp -o ConnectTimeout=15 "${SSH_ALIAS}:/home/ec2-user/app/billiard_turns_export.jsonl" "$EXPORT" 2>/dev/null; then
  echo "  -> 최신본 수신: $(wc -l < "$EXPORT" | tr -d ' ') 레코드"
else
  echo "  ! 서버 접속 실패 — 기존 로컬 export 로 진행 ($( (wc -l < "$EXPORT" 2>/dev/null || echo 0) | tr -d ' ') 레코드)"
fi

# --- 2) 경기 전 승률/선수 검색용 RDS 터널 (캠퍼스 네트워크는 RDS:5432 직접 접속이 막혀 있음) ---
#     비밀번호는 여기 없다 — DATABASE_URL은 각자 로컬의 YOLO/.env 에서 dotenv 로 읽는다.
echo "[2/3] RDS 접속 터널 확인 중 (localhost:${TUNNEL_PORT}) ..."
TUNNEL_PID=""
if lsof -iTCP:"$TUNNEL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  -> 이미 열려 있음, 재사용"
elif ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$SSH_ALIAS" true 2>/dev/null; then
  echo "  ! '${SSH_ALIAS}' SSH 접속 불가 — DB 없이 진행 (경기 전 승률/선수 검색 비활성)"
else
  ssh -N -L "${TUNNEL_PORT}:${RDS_HOST}:5432" \
    -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$SSH_ALIAS" &
  TUNNEL_PID=$!
  for _ in $(seq 1 20); do
    lsof -iTCP:"$TUNNEL_PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
    sleep 0.25
  done
  if lsof -iTCP:"$TUNNEL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  -> 터널 열림"
  else
    echo "  ! 터널이 열리지 않음 — DB 없이 진행"
    TUNNEL_PID=""
  fi
fi
trap 'if [ -n "$TUNNEL_PID" ]; then kill "$TUNNEL_PID" 2>/dev/null || true; fi' EXIT

# --- 3) CueCast 기동 (CatBoost 학습에 수십 초) ---
echo "[3/3] CueCast 기동 중 -> http://127.0.0.1:8765   (Ctrl+C 로 종료)"
"$PY" YOLO/local_probability_server.py --shots "$EXPORT" --host 127.0.0.1 --port 8765
