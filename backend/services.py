# 데이터 조회·분석 실행 (= Spring 의 Service 계층)
#
# 주의: 맥(캠퍼스망)에서는 RDS(5432)에 직접 못 붙는다. 그래서 턴 데이터는
# ① 로컬 results/<id>/turns.jsonl → ② 없으면 S3(443, 통과됨) 순서로 읽는다.
# 백엔드를 EC2로 배포하면 그때 DB 직조회로 바꾸면 된다.
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PENDING = os.path.join(ROOT, "jobs", "pending")
PYTHON = os.path.join(ROOT, "venv", "bin", "python")
DB_ENV = os.path.join(ROOT, "db", "db.env")


def _env_from_db_env() -> dict:
    """db/db.env 의 `export KEY="value"` 들을 파싱해 환경변수 dict 로 반환."""
    env = dict(os.environ)
    if os.path.exists(DB_ENV):
        for line in open(DB_ENV):
            m = re.match(r'\s*export\s+(\w+)="?([^"\n]*)"?', line)
            if m:
                env[m.group(1)] = m.group(2)
    return env


def video_id_of(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else re.sub(r"\W", "_", url)[-20:]


# ---------- 턴 데이터 조회 ----------
def load_turns(video_id: str) -> list[dict]:
    """로컬 results → 없으면 S3 에서 turns.jsonl 로드."""
    path = os.path.join(RESULTS, video_id, "turns.jsonl")
    if not os.path.exists(path):
        _try_fetch_from_s3(video_id, path)
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def _try_fetch_from_s3(video_id: str, dest: str) -> None:
    env = _env_from_db_env()
    bucket = env.get("S3_BUCKET", "").replace("s3://", "").strip()
    if not bucket:
        return
    try:
        import boto3
        s3 = boto3.client("s3", region_name=env.get("AWS_REGION") or None)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        s3.download_file(bucket, f"results/{video_id}/turns.jsonl", dest)
    except Exception:
        pass  # S3에도 없으면 조용히 빈 결과 (분석 전인 영상)


def list_videos() -> list[dict]:
    """분석된 영상 목록 (로컬 results 기준)."""
    out = []
    if os.path.isdir(RESULTS):
        for vid in sorted(os.listdir(RESULTS)):
            p = os.path.join(RESULTS, vid, "turns.jsonl")
            if os.path.exists(p):
                n = sum(1 for line in open(p) if line.strip())
                out.append({"video_id": vid, "turns": n})
    return out


def turn_at(video_id: str, t: float) -> dict | None:
    """재생시간 t초가 속한 턴 (프론트 동기화용).
    샷 구간 사이(다음 샷 준비 중)면 직전 턴을 돌려준다."""
    prev = None
    for rec in load_turns(video_id):
        if rec["time_start_s"] <= t <= rec["time_end_s"]:
            return rec
        if rec["time_end_s"] < t:
            prev = rec
        else:
            break
    return prev


# ---------- 분석 실행 ----------
def enqueue_analysis(url: str) -> str:
    """URL 을 큐에 넣고 워커(process_queue)를 백그라운드로 기동. video_id 반환."""
    vid = video_id_of(url)
    os.makedirs(PENDING, exist_ok=True)
    job = os.path.join(PENDING, f"api_{int(time.time())}_{vid}.txt")
    with open(job, "w") as f:
        f.write(url + "\n")
    # 워커는 자체 락이 있어 중복 기동해도 안전. S3 업로드용 env 를 함께 전달.
    subprocess.Popen(
        [PYTHON, os.path.join(ROOT, "src", "process_queue.py")],
        env=_env_from_db_env(), cwd=ROOT,
        stdout=open(os.path.join(ROOT, "logs_api_worker.txt"), "a"),
        stderr=subprocess.STDOUT,
    )
    return vid

# ---------- 분석 상태 ----------
RUNNING = os.path.join(ROOT, "jobs", "running")
FAILED = os.path.join(ROOT, "jobs", "failed")

def _queue_contains(folder: str, video_id: str) -> bool:
    """큐 폴더 안의 작업 파일(.txt)들 중 이 video_id 가 든 URL 이 있는지."""
    if not os.path.isdir(folder):
        return False
    for fn in os.listdir(folder):
        if not fn.endswith(".txt"):
            continue
        try:
            if video_id in open(os.path.join(folder, fn)).read():
                return True
        except OSError:
            pass
    return False

def get_status(video_id: str) -> dict:
    """분석 진행 상태. 프론트가 /analyze 후 폴링하는 용도.
    우선순위: done(결과 있음) > running > queued > failed > not_found"""
    turns = load_turns(video_id)          # 로컬 → S3 폴백까지 포함
    if turns:
        return {"video_id": video_id, "state": "done", "turns": len(turns)}
    if _queue_contains(RUNNING, video_id):
        return {"video_id": video_id, "state": "running", "turns": 0}
    if _queue_contains(PENDING, video_id):
        return {"video_id": video_id, "state": "queued", "turns": 0}
    if _queue_contains(FAILED, video_id):
        return {"video_id": video_id, "state": "failed", "turns": 0}
    return {"video_id": video_id, "state": "not_found", "turns": 0}

# ---------- 확률 엔진 연동 (팀원 CueCast 서버 호출) ----------
import urllib.request

PROB_SERVER = os.environ.get("PROB_SERVER", "http://127.0.0.1:8765")


def shot_probability(shooter: str, before: dict) -> dict | None:
    """팀원 확률 서버에 배치를 보내 성공 확률을 받는다. 서버가 꺼져 있으면 None."""
    body = json.dumps({"shooter": shooter, "before": before}).encode()
    req = urllib.request.Request(
        f"{PROB_SERVER}/api/v1/shot-probability",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            d = json.loads(resp.read())
        return {
            "successProbability": d.get("successProbability"),
            "modelVersion": d.get("modelVersion"),
            "components": d.get("components"),
        }
    except Exception:
        return None   # 엔진 다운이어도 턴 데이터는 정상 응답 (확률만 빠짐)