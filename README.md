# 26s-w3-c3-01
몰입캠프 26s-w3-c3-01 프로젝트 repository

## 당구 영상 → 턴 데이터 자동 추출 파이프라인

유튜브 3쿠션 중계 영상에서 매 턴(샷)의 데이터를 자동으로 뽑아 서버 DB(PostgreSQL)에 저장한다:

- `before`: 샷 직전 공 3개 위치 (당구대 기준 0~1 정규화, [x, y])
- `shooter`: 수구 (white / yellow) — 정지 배치에서 가장 먼저 움직인 공
- `success`: 3쿠션 성공 여부 — 수구 궤적에서 "두 목적구 접촉 + 두 번째 접촉 전 쿠션 3회"를 직접 계산
- `after`: 샷 이후 공 3개 위치 (`after_source: settled`=정지 확인, `last_seen`=영상 컷으로 마지막 관측값)

---

## 실행 흐름 (전체 구조)

연산은 **맥에서**(YOLO, GPU 무료), 저장은 **AWS에서**. 맥은 S3에 올리기만 하고, DB 적재는 AWS 안에서 자동으로 처리된다.

```
[내 맥 · 캠퍼스망]                              [AWS 서울]

 ① jobs/pending/*.txt 에 URL 등록
 ② src/process_queue.py 실행
    ├─ 유튜브 다운로드 (yt-dlp)
    ├─ src/extract_turns.py 로 턴 추출
    └─ 결과 S3 업로드 ──(443, 통과)──▶  S3: billiard-turns-mcamp3w/results/<영상>/
                                                  │
                                        ③ EC2 로더 cron (10분마다 · 자동)
                                           load_to_db.py --all
                                                  ▼
                                        RDS PostgreSQL: billiard_turns  ← "서버 DB"
                                                  │
                                        ④ (선택) db/export_db.py → S3 exports/ → 맥으로 scp
```

- 맥에서 **②까지만 실행**하면, S3 → DB 적재(③)는 EC2가 10분 이내에 **자동**으로 한다.
- 캠퍼스망이 AWS의 DB 포트(5432)를 막기 때문에, 맥은 **S3(443)에만** 올리고 실제 DB 적재는 **같은 VPC 안의 EC2**가 담당한다.

---

## 실행 방법

### A. 유튜브 URL → 서버 DB (기본 사용법)

```bash
cd ~/26s-w3-c3-01

# 1) URL을 큐에 등록 (한 줄에 링크 1개 · 여러 줄/여러 파일 가능 · '#'은 주석)
echo "https://www.youtube.com/watch?v=XXXX" > jobs/pending/myjob.txt

# 2) 워커 실행 (다운로드 → 추출 → S3 업로드)
source db/db.env && venv/bin/python src/process_queue.py
```

- `source db/db.env` 로 `S3_BUCKET`·`AWS_REGION` 을 불러와야 **S3까지 업로드**된다. 안 하면 로컬(`results/`)에만 저장된다.
- 업로드가 끝나면 **10분 이내** EC2 cron이 알아서 DB에 적재한다. 여기까지가 끝.
- 추출 후 다운로드한 원본 영상은 자동 삭제(디스크 절약). 남기려면 앞에 `KEEP_VIDEOS=1` 을 붙인다.

### B. 파이프라인 점검 (전 구간 한 번에 확인)

```bash
bash deploy/verify_pipeline.sh
```

로컬→EC2 SSH → S3 권한 → cron → RDS 저장 현황 → export 파일까지 순서대로 확인한다.
(`~/.ssh/config` 에 `Host billiard` 별칭이 설정돼 있어야 한다.)

### C. 단일 영상만 추출 (S3/DB 없이 로컬에서)

```bash
venv/bin/python src/extract_turns.py videos/영상.mp4 --outdir results/영상ID --save-frames
```

출력: `results/영상ID/turns.jsonl`, `turns.csv`, (옵션) `qa/` 턴별 검증용 프레임.

### D. (선택) 맥에서 큐 자동 처리 — crontab

새 URL을 넣어두면 주기적으로 알아서 처리하게 하려면(10분마다, 이중 실행은 락으로 방지):

```
*/10 * * * * cd /Users/parkminsu/26s-w3-c3-01 && source db/db.env && venv/bin/python src/process_queue.py >> logs/worker.log 2>&1
```

**필요 도구:** `venv/bin/yt-dlp`, JS 런타임(node)이 PATH에, 로컬 AWS 자격증명(`~/.aws/credentials`).

---

## 파일 구성

엔진 코드는 `src/` 에, 실행·유틸(DB·배포)은 각 폴더에 있다.

| 파일 (`src/`) | 역할 |
|---|---|
| `detect_pipeline.py` | 당구대 꼭짓점 검출 + 픽셀→정규화 좌표 변환 |
| `detect_video.py` | 프레임별 탐지/미니맵 합성 영상 + 시뮬레이션 확률 (시각화용) |
| `extract_turns.py` | 턴 단위 데이터 추출 (배치/서버용 핵심 모듈) |
| `process_queue.py` | 유튜브 링크 큐 워커 (다운로드 → 추출 → S3 업로드) |
| `simulate.py` | 3쿠션 물리 시뮬레이션 (성공 확률 추정) |
| `best_3cls.pt` | YOLO 공 탐지 모델 (white/yellow/red) |

| 파일 (`db/` · `deploy/`) | 역할 |
|---|---|
| `db/load_to_db.py` | S3 → RDS PostgreSQL upsert (EC2 cron이 10분마다 실행) |
| `db/export_db.py` | RDS → S3 `exports/` 내보내기 (맥이 결과 회수용) |
| `db/schema.sql` | `billiard_turns` 테이블 스키마 |
| `deploy/verify_pipeline.sh` | 로컬→S3→RDS 전 구간 점검 스크립트 |
| `deploy/iam-billiard-loader-role.json` | 로더 역할 S3 읽기/쓰기 IAM 정책 |

---

## turns.jsonl 스키마 예시

```json
{"video_id": "WV3tL6z3cqo", "turn": 3, "epoch": 2, "shooter": "yellow",
 "before": {"white": [0.42, 0.31], "yellow": [0.38, 0.62], "red": [0.71, 0.28]},
 "after":  {"white": [0.15, 0.44], "yellow": [0.55, 0.80], "red": [0.61, 0.22]},
 "after_source": "settled",
 "success": true,
 "success_detail": {"method": "trajectory", "coverage": 0.87,
                    "hits": ["red", "white"], "cushions_before_2nd": 4},
 "frame_start": 1520, "frame_end": 1893,
 "time_start_s": 50.7, "time_end_s": 63.1}
```

- `epoch`: 탑뷰가 길게 끊길 때마다 증가하는 클립 번호. 같은 epoch 안의 연속 턴만 시간적으로 이어진 것.
- `success: null` = 궤적 관측이 부족해 판정 보류 (`success_detail.method: "insufficient"`).
