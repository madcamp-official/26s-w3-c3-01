# GPU 인스턴스 자동 실행 배포 가이드

유튜브 링크를 큐에 넣으면 → 꺼져 있던 GPU 인스턴스가 켜지고 → 데이터를 뽑고 →
할 일이 없어지면 스스로 정지한다. 처리하는 동안에만 GPU 요금이 나간다.

## 구성

```
로컬(노트북)                     EC2 GPU 인스턴스 (평소엔 stopped)
────────────                     ─────────────────────────────
submit_job.sh  ──(start+SSH)──▶  jobs/pending/ 에 링크 등록
                                 cron: worker_autostop.sh (3분마다)
                                   ├ process_queue.py 로 다운로드+추출
                                   └ 큐가 비면 자기 자신을 stop
                                 결과: results/<video_id>/turns.jsonl
```

## 1회 셋업

### A. EC2 인스턴스 (아래 "추천 사양" 참고)
1. **Deep Learning AMI (Ubuntu, PyTorch/CUDA 사전설치)** 로 인스턴스 생성.
2. IAM 역할을 붙여 **자기 자신을 정지**(`ec2:StopInstances`)하고 **결과를 S3에 업로드**
   (`s3:PutObject`, `s3:ListBucket` — 해당 버킷 대상)할 수 있게 한다.
3. 레포를 인스턴스에 올리고, CUDA용 torch를 설치:
   ```bash
   cd ~/26s-w3-c3-01
   python3 -m venv venv
   venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   venv/bin/pip install ultralytics opencv-python-headless yt-dlp
   venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"  # True 여야 함
   ```
4. yt-dlp용 JS 런타임(node) 설치: `sudo apt-get install -y nodejs`
5. 설정 파일 작성: `cp deploy/config.sh.example deploy/config.sh` 후 값 채우기.
6. cron 등록:
   ```bash
   crontab -e
   */3 * * * * /home/ubuntu/26s-w3-c3-01/deploy/worker_autostop.sh >> /home/ubuntu/26s-w3-c3-01/logs/worker_autostop.log 2>&1
   ```
7. 인스턴스를 **정지**시켜 둔다 (평소 상태).

### B. 로컬(노트북)
- AWS CLI 설치 + `aws configure` 로 자격증명 등록 (권한: `ec2:StartInstances`, `ec2:DescribeInstances`).
- SSH 키(.pem)를 config.sh 의 `SSH_KEY` 경로에 둔다.
- `cp deploy/config.sh.example deploy/config.sh` 후 동일하게 값 채우기.

## 사용법

```bash
# 링크 하나
./deploy/submit_job.sh "https://www.youtube.com/watch?v=XXXX"

# 여러 개 한 번에
./deploy/submit_job.sh "https://youtu.be/AAAA" "https://youtu.be/BBBB"
```

넣고 나면 끝. 인스턴스가 알아서 켜지고, 처리하고, 정지한다.

## 결과 저장 · 영상 정리 (자동)

추출이 성공하면 `process_queue.py` 가 자동으로:
1. 결과 폴더 전체(`turns.jsonl`, `turns.csv`, `traj.json`, `qa/`)를
   **S3로 업로드** — `s3://<S3_BUCKET>/<S3_PREFIX>/<video_id>/` 경로.
2. **원본 영상 삭제** (수 GB, 재다운로드 가능하므로) — 디스크가 안 찬다.

업로드가 실패하면 그 작업은 `jobs/failed/` 로 가고 영상도 남아, 재시도 시 재다운로드가 필요 없다.
`config.sh` 의 `S3_BUCKET` 을 비우면 S3 업로드를 건너뛰고 로컬에만 저장한다(로컬 테스트용).

### 왜 DB가 아니라 S3인가
워커 인스턴스는 켜졌다 꺼지는 임시 환경이라, DB에 직접 넣으면 매 실행마다 DB 접속·스키마·
자격증명에 의존한다. 대신 결과를 S3에 durable 하게 쌓아두고, **DB 적재는 별도 로더가
S3를 읽어 처리**하면 결합도가 낮고 재처리·백필이 쉽다.

## S3 → PostgreSQL 적재 (db/load_to_db.py)

S3에 쌓인 결과를 PostgreSQL 로 옮기는 로더. GPU 워커와 **분리해서** DB 근처(예: 상시 켜둔
작은 인스턴스나 로컬)에서 실행한다. `(video_id, turn)` 자연키로 **upsert** 하므로 몇 번을
재실행해도 중복이 안 생긴다(idempotent) — cron 으로 주기 실행하기 좋다.

```bash
pip install -r db/requirements.txt          # psycopg2-binary, boto3

# S3 prefix 아래 모든 영상 적재 (스키마는 최초 실행 시 자동 생성)
DATABASE_URL=postgres://user:pw@host:5432/db S3_BUCKET=s3://my-bucket \
  python db/load_to_db.py --all

# 특정 영상 하나만
python db/load_to_db.py --video-id WV3tL6z3cqo   # (DATABASE_URL, S3_BUCKET 환경변수 사용)

# S3 없이 로컬 results/ 에서 (테스트용)
DATABASE_URL=... python db/load_to_db.py --all --local results
```

cron 예시 (10분마다 S3 → DB 동기화):
```
*/10 * * * * DATABASE_URL=... S3_BUCKET=s3://my-bucket /path/venv/bin/python /path/db/load_to_db.py --all >> /path/logs/loader.log 2>&1
```

테이블은 `billiard_turns` (턴별 데이터) + `billiard_ingest_log` (영상별 적재 이력). 스키마는
`db/schema.sql` 참고. 좌표는 `before_pos`/`after_pos` JSONB 컬럼이라 이렇게 질의한다:
```sql
-- 백구가 수구였고 성공한 턴의 수구 시작 위치
SELECT video_id, turn, before_pos->'white' AS white_start
FROM billiard_turns WHERE shooter='white' AND success;
```

## 동작 세부

- **중복 처리 안 함**: `process_queue.py` 의 mkdir 락으로, cron이 겹쳐 떠도 실제 처리는 하나만 돈다.
- **작업 중 정지 안 함**: 유휴 판정은 `pending` + `running` 이 **둘 다** 비었을 때만. 처리 중인 작업은 `running/` 에 있어 정지가 막힌다.
- **바로 안 끔**: `IDLE_CYCLES_BEFORE_STOP` 만큼(기본 2회 = 약 6분) 연속으로 할 일이 없어야 정지. 방금 끝난 직후 새 작업이 들어올 여지를 준다.

## 운영 주의

- **디스크**: 추출 성공 시 원본 영상을 자동 삭제하므로 `videos/` 는 안 쌓인다. 단, 처리 중인
  영상 1개는 있으니 EBS 는 50GB+ 권장(AMI 자체 용량 포함해 100GB 잡으면 넉넉).
- **결과 보존**: 성공한 결과는 S3에 올라가므로 인스턴스를 종료(terminate)해도 안전하다.
  단 `S3_BUCKET` 을 반드시 설정해 둘 것 — 안 하면 로컬 EBS 에만 남아 종료 시 사라진다.
- **정지 vs 종료**: 이 스크립트는 stop(정지)만 한다. 정지 상태에선 EBS 요금(소액)만 나가고 GPU 요금은 0.
