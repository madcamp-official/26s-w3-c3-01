# 26s-w3-c3-01
몰입캠프 26s-w3-c3-01 프로젝트 repository

## 당구 영상 → 턴 데이터 자동 추출 파이프라인

유튜브 3쿠션 중계 영상에서 매 턴(샷)의 데이터를 자동으로 뽑아 서버 DB(PostgreSQL)에 저장한다.
**턴은 방송 점수판의 이닝 이벤트가 정의하고**(영상 추적이 끊겨도 점수판 변화만 있으면 턴 확정),
영상(YOLO)은 그 턴의 공 좌표만 채운다. 상세 판정 기준: [docs/EXTRACTION_CRITERIA.md](docs/EXTRACTION_CRITERIA.md).

- `before`: 샷 직전 공 3개 위치 (당구대 기준 0~1 정규화, [x, y])
- `shooter`: 수구 (white / yellow) — **점수판 원형(현재 이닝 표시) 기준**: 점수 박스 오른쪽
  원형에 숫자가 떠 있는 색이 지금 치는 선수다 (턴 교대 시 상대 원형에 0이 새로 나타남).
  원형을 못 읽는 구간만 "가장 먼저 움직인 공" 추정으로 폴백 (`success_detail.shooter_source` 로 구분)
- `success`: 3쿠션 성공 여부 — **방송 점수판 OCR 전용**: 샷 구간에 수구 색 점수가 +1(+2=뱅크샷)
  오르면 성공, 끝까지 안 오르면 실패. 점수판은 오퍼레이터가 올리는 사실상의 정답이다.
  (PBA 점수판은 흰 박스=흰 수구 선수, 노란 박스=노란 수구 선수라 수구와 바로 매칭)
  **점수판으로 판정 못 한 턴은 폐기한다** — 궤적(쿠션 세기) 판정은 신뢰도가 낮아 라벨로
  쓰지 않는다. 점수판이 없는 영상(스포방지 마스킹·비방송 영상)은 0턴이 된다.
  DB의 `success_method` 는 항상 `scoreboard`, 로더도 그 외 판정을 걸러낸다(이중 방어).
- `after`: 샷 이후 공 3개 위치. `after_source`로 좌표 출처 표시 — `still`/`still_near`(정지 배치·정확),
  `obs_near`/`obs_far`(공별 최근접 관측·근사). 턴 N의 `after` = 턴 N+1의 `before` (연결됨)
- `bank_shot`: 뱅크샷(+2점) 여부 — 이닝 원형 +2 또는 총점 박스 +2 교차 판정
- `coverage`: 탑뷰 관측 비율 — 낮으면 좌표가 근사치(라벨은 점수판이라 여전히 유효)

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
- **이미 추출한 영상은 자동으로 건너뛴다** — S3(또는 로컬)에 `results/<video_id>/turns.jsonl` 이 있으면 그 URL은 스킵. 같은 링크를 다시 넣어도 재다운로드·재추출을 안 한다. 강제로 다시 추출하려면 앞에 `FORCE=1` 을 붙인다.
- **실행 이력은 `jobs/done/processed_urls.txt` 에 누적된다** — 링크마다 `시각 상태(done/skip/failed) URL` 한 줄. 작업 파일 자체도 처리 후 `jobs/done/`(실패 시 `jobs/failed/`)로 이동한다.
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

**필요 도구:** `venv/bin/yt-dlp`, JS 런타임(node)이 PATH에, 로컬 AWS 자격증명(`~/.aws/credentials`),
점수판 OCR용 `tesseract` (`brew install tesseract` — 없으면 궤적 판정으로 자동 폴백).

---

## 파일 구성

엔진 코드는 `src/` 에, 실행·유틸(DB·배포)은 각 폴더에 있다.

| 파일 (`src/`) | 역할 |
|---|---|
| `detect_pipeline.py` | 당구대 꼭짓점 검출 + 픽셀→정규화 좌표 변환 |
| `detect_video.py` | 프레임별 탐지/미니맵 합성 영상 + 시뮬레이션 확률 (시각화용) |
| `extract_turns.py` | 턴 단위 데이터 추출 (배치/서버용 핵심 모듈) |
| `scoreboard.py` | 방송 점수판 OCR (박스 자동 탐지 + 점수 변화 이벤트) |
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
{"video_id": "당구분석3", "turn": 2, "epoch": 0, "shooter": "yellow",
 "before": {"white": [0.66, 0.24], "yellow": [0.68, 0.29], "red": [0.11, 0.13]},
 "after":  {"white": [0.62, 0.44], "yellow": [0.34, 0.41], "red": [0.11, 0.94]},
 "after_source": "obs_near",
 "success": true,
 "success_detail": {"method": "scoreboard", "shooter_source": "scoreboard",
                    "run_from": 1, "run_to": 3, "total_delta": 2,
                    "totals": [[6, 4], [6, 6]], "bank_shot": true, "coverage": 0.59,
                    "before_source": "still_near", "after_pos_source": "obs_near",
                    "hits": [], "cushions_before_2nd": null},
 "frame_start": 280, "frame_end": 1290,
 "time_start_s": 4.9, "time_end_s": 22.9}
```

- `epoch`: 탑뷰가 길게 끊길 때마다 증가하는 클립 번호. 같은 epoch 안의 연속 턴만 시간적으로 이어진 것.
- `score_steps`: 점수판 판정 근거 — 샷 창에서 관측된 점수 변화 `[프레임, 흰Δ, 노랑Δ]` 목록.
- `bank_shot`: 점수가 한 번에 +2 오른 성공 = 뱅크샷(3쿠션 선행 후 두 목적구 접촉, PBA 2점).
  점수판 판정 턴에만 채워짐 (궤적 판정 턴은 DB에서 `NULL`).
- `traj_success`: 점수판 판정으로 덮어쓰기 전 궤적 판정 결과 (두 판정 비교·QA용).
