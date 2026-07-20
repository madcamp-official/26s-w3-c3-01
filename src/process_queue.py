# 유튜브 링크 큐 워커: jobs/pending/의 링크를 집어 다운로드 → 턴 추출 → 결과 저장.
# crontab 등록 예시 (10분마다):
#   */10 * * * * cd /Users/parkminsu/26s-w3-c3-01 && venv/bin/python process_queue.py >> logs/worker.log 2>&1
# 작업 추가:
#   echo "https://www.youtube.com/watch?v=XXXX" > jobs/pending/작업이름.txt  (한 줄에 링크 1개, 여러 줄 가능)
# 결과: results/<video_id>/turns.jsonl, turns.csv
#
# 재처리 방지: 이미 추출된 영상(= S3 또는 로컬 results 에 turns.jsonl 존재)은 자동으로 건너뛴다.
# 다시 추출하고 싶으면 FORCE=1 로 실행:  FORCE=1 source db/db.env && venv/bin/python src/process_queue.py
# 실행 이력: 처리한 모든 링크가 jobs/done/processed_urls.txt 에 누적 기록된다 (시각·상태·URL).
import os
import re
import shutil
import subprocess
import sys
import time

# 이 파일은 src/ 안에 있으므로, 프로젝트 루트는 그 부모 폴더 (jobs/videos/results/venv 등이 여기 있음)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PENDING = os.path.join(ROOT, "jobs", "pending")
RUNNING = os.path.join(ROOT, "jobs", "running")
DONE = os.path.join(ROOT, "jobs", "done")
FAILED = os.path.join(ROOT, "jobs", "failed")
VIDEOS = os.path.join(ROOT, "videos")
RESULTS = os.path.join(ROOT, "results")
LOCK = os.path.join(ROOT, "jobs", ".worker.lock")
PYTHON = os.path.join(ROOT, "venv", "bin", "python")
YTDLP = os.path.join(ROOT, "venv", "bin", "yt-dlp")
# h264 mp4 영상만(오디오 불필요, OpenCV 호환) 1080p 이하 최고 화질
YTDLP_FORMAT = "bv*[ext=mp4][vcodec^=avc1][height<=1080]/bv*[height<=1080]/b"
STALE_LOCK_S = 6 * 3600

# 결과 저장(S3) / 원본 영상 정리 설정 — cron에서는 worker_autostop.sh 가 config.sh 를
# source 하므로 거기서 export 한 값이 여기로 들어온다. 값이 없으면 각 단계를 건너뛴다.
S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()       # 예: s3://my-bucket 또는 my-bucket
S3_PREFIX = os.environ.get("S3_PREFIX", "results").strip("/")
KEEP_VIDEOS = os.environ.get("KEEP_VIDEOS") == "1"        # 1이면 추출 후에도 원본 영상 보존


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def upload_results(vid, outdir):
    """결과 폴더 전체(turns.jsonl/csv, traj.json, qa/)를 S3에 업로드 (boto3).
    aws CLI 불필요 — 캠퍼스망에서 맥은 이 업로드(HTTPS 443)만 하면 된다.
    S3_BUCKET 이 설정돼 있으면 필수 단계로 취급 — 실패 시 예외를 올려 작업을 failed 처리한다."""
    if not S3_BUCKET:
        log("S3_BUCKET 미설정 → S3 업로드 건너뜀 (로컬에만 저장)")
        return
    import boto3
    bucket = S3_BUCKET[len("s3://"):] if S3_BUCKET.startswith("s3://") else S3_BUCKET
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or None)
    n = 0
    for root, _, files in os.walk(outdir):
        for fn in files:
            local = os.path.join(root, fn)
            key = f"{S3_PREFIX}/{vid}/{os.path.relpath(local, outdir)}"
            s3.upload_file(local, bucket, key)
            n += 1
    log(f"S3 업로드: {outdir} → s3://{bucket}/{S3_PREFIX}/{vid}/ ({n}개 파일)")


def video_id_of(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else re.sub(r"\W", "_", url)[-20:]


def already_processed(vid):
    """이미 추출된 영상인지 판정해 재처리를 막는다.
    판정 우선순위: (1) S3 에 results/<vid>/turns.jsonl 존재  (2) 로컬 results/<vid>/turns.jsonl 존재.
    FORCE=1 이면 항상 False 를 반환해 강제로 재처리한다."""
    if os.environ.get("FORCE") == "1":
        return False
    if S3_BUCKET:
        try:
            import boto3
            bucket = S3_BUCKET[len("s3://"):] if S3_BUCKET.startswith("s3://") else S3_BUCKET
            s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or None)
            s3.head_object(Bucket=bucket, Key=f"{S3_PREFIX}/{vid}/turns.jsonl")
            return True  # S3에 결과가 있음 = DB에 반영됐거나 다음 cron에 반영될 것
        except Exception:
            pass  # 없음(404)이거나 조회 실패 → 로컬 확인으로 폴백
    return os.path.exists(os.path.join(RESULTS, vid, "turns.jsonl"))


def process_url(url):
    """URL 하나 처리. 반환: 'done'(처리됨) | 'skip'(이미 처리돼 건너뜀)."""
    vid = video_id_of(url)
    if already_processed(vid):
        log(f"이미 처리된 영상 — 건너뜀: {vid}  ({url})   [재처리하려면 FORCE=1]")
        return "skip"
    video_path = os.path.join(VIDEOS, f"{vid}.mp4")
    outdir = os.path.join(RESULTS, vid)

    if not os.path.exists(video_path):
        log(f"다운로드: {url} → {video_path}")
        subprocess.run(
            [YTDLP, "--js-runtimes", "node", "-f", YTDLP_FORMAT,
             "-o", video_path, url],
            check=True, timeout=1800)

    log(f"턴 추출: {video_path}")
    subprocess.run(
        [PYTHON, os.path.join(SRC, "extract_turns.py"), video_path,
         "--video-id", vid, "--outdir", outdir, "--save-frames"],
        check=True, cwd=ROOT, timeout=4 * 3600)
    log(f"완료: {outdir}/turns.jsonl")

    # 결과를 S3로 내보낸 뒤(성공 시) 원본 영상 삭제. 순서가 중요하다: 업로드가
    # 실패하면 여기서 예외가 올라가 영상이 남으므로, 재시도 시 재다운로드가 필요 없다.
    upload_results(vid, outdir)

    if not KEEP_VIDEOS and os.path.exists(video_path):
        os.remove(video_path)
        log(f"원본 영상 삭제: {video_path}")
    return "done"


def record_ledger(url, status):
    """실행한 링크를 누적 이력 파일(jobs/done/processed_urls.txt)에 한 줄씩 기록.
    작업 파일(pending→done 이동)과 별개로, 언제 어떤 링크가 어떻게 끝났는지 남긴다."""
    ledger = os.path.join(DONE, "processed_urls.txt")
    with open(ledger, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {status:6s}  {url}\n")


def main():
    for d in (PENDING, RUNNING, DONE, FAILED, VIDEOS, RESULTS,
              os.path.join(ROOT, "logs")):
        os.makedirs(d, exist_ok=True)

    # 이중 실행 방지 락 (mkdir는 원자적; 오래된 락은 죽은 워커로 보고 제거)
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) > STALE_LOCK_S:
        os.rmdir(LOCK)
    try:
        os.mkdir(LOCK)
    except FileExistsError:
        log("다른 워커가 실행 중 — 종료")
        return
    try:
        jobs = sorted(f for f in os.listdir(PENDING) if f.endswith(".txt"))
        if not jobs:
            return
        for job in jobs:
            src = os.path.join(PENDING, job)
            run = os.path.join(RUNNING, job)
            shutil.move(src, run)
            ok = True
            with open(run) as f:
                urls = [u.strip() for u in f if u.strip() and not u.startswith("#")]
            for url in urls:
                try:
                    status = process_url(url)
                except Exception as e:
                    ok = False
                    status = "failed"
                    log(f"실패: {url} — {e}")
                record_ledger(url, status)
            shutil.move(run, os.path.join(DONE if ok else FAILED, job))
            log(f"작업 {job} → {'done' if ok else 'failed'}")
    finally:
        os.rmdir(LOCK)


if __name__ == "__main__":
    main()
