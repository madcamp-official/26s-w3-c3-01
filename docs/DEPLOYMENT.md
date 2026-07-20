# 🎱 CueCast 시스템 전체 구성 & 배포 가이드

> 유튜브 3쿠션 영상에서 **턴 데이터(공 위치·수구·성공)를 자동 수집**하고,
> 그 데이터로 학습한 **하이브리드 확률 엔진(CueCast)** 을 웹으로 서비스하는 시스템.
> 처음 보는 사람은 이 문서 하나로 구조·운영·사용법을 파악할 수 있다.

---

## 1. 한눈에 보는 전체 구조

```
[내 맥 · 캠퍼스망]                                [AWS 서울]
─────────────────                                ─────────────────────────────────────
 ① 유튜브 URL 큐 등록                              S3 (billiard-turns-mcamp3w)
 ② src/process_queue.py 실행                        ├ results/<영상>/turns.jsonl  ← 맥이 업로드
    ├ yt-dlp 다운로드                               └ exports/billiard_turns.jsonl
    ├ src/extract_turns.py 로 턴 추출                        ▲
    └ 결과 S3 업로드 ──(443, 캠퍼스망 통과)──▶              │
                                                  EC2 t3.small (billiard-loader)
                                                    ├ cron: S3 → RDS 적재      (10분마다)
                                                    ├ cron: RDS → export 갱신  (10분마다)
                                                    ├ CueCast 서버 (8765)  ─┐
                                                    ├ FastAPI  서버 (8000) ─┤ cloudflared 터널
                                                    └ cron: 매일 06시 재시작  │
                                                             ▼               ▼
                                                  RDS PostgreSQL      https://xxxx.trycloudflare.com
                                                  (billiard_turns)    (어디서든 브라우저 접속)
```

**핵심 설계 이유**
- 캠퍼스망이 AWS로 가는 **5432(DB)·22(SSH) 포트를 차단** → 맥은 **S3(443)에만** 올리고,
  DB 작업은 전부 **같은 VPC 안의 EC2**가 담당한다.
- **연산(YOLO 추출)은 맥**(Apple GPU, 무료)에서, **서비스·저장은 AWS**에서.
- 외부 접속은 IP/보안그룹 대신 **cloudflared 터널**(443, 캠퍼스망 통과·IP 무관)로.

---

## 2. 구성 요소별 설명

### 데이터 수집 파이프라인 (맥, `src/`)
| 파일 | 역할 |
|---|---|
| `src/process_queue.py` | 큐 워커: 다운로드 → 추출 → S3 업로드 → 원본 삭제 (락으로 중복 방지) |
| `src/extract_turns.py` | 턴 추출: 정지 배치 감지 → 수구 식별 → 점수판 기반 성공 판정 |
| `src/detect_pipeline.py` | 당구대 검출 (파랑·초록·회색 천 3종 방송 자동 대응) |
| `src/detect_video.py` | 탑뷰(부감 카메라) 판별 |
| `src/best_3cls.pt` | YOLO 공 탐지 모델 (white/yellow/red) |

### DB·적재 (`db/`, EC2에서 cron 실행)
| 파일 | 역할 |
|---|---|
| `db/load_to_db.py` | S3의 turns.jsonl → RDS upsert. **점수판(scoreboard) 판정 턴만 적재** |
| `db/export_db.py` | RDS 전체 → `billiard_turns_export.jsonl` (CueCast 학습용) + S3 exports/ |
| `db/schema.sql` | `billiard_turns` 테이블 (좌표는 0~1 정규화 JSONB) |

### 확률 엔진 + UI (팀원 작업, `YOLO/`)
| 파일 | 역할 |
|---|---|
| `YOLO/local_probability_server.py` | **CueCast 서버**: 웹 UI + 하이브리드 확률 엔진(CatBoost+Neighbor+Grid) + 유튜브 분석 |
| `YOLO/ui/` | 웹 화면 (실시간 분석 / 샷 기록 / 선수 통계) |
| `YOLO/extension/` | 크롬 확장(사이드패널) 버전 |
- 시작 시 `--shots` 파일(= DB export)로 **CatBoost 모델을 학습**하고 서비스한다.

### 조회 API (`backend/`, FastAPI)
| 엔드포인트 | 역할 |
|---|---|
| `GET /videos` | 분석된 영상 목록 |
| `GET /videos/{id}/turns` | 영상 전체 턴 데이터 |
| `GET /videos/{id}/turn-at?time=초` | 재생시간에 해당하는 턴 (+확률 결합) |
| `GET /videos/{id}/status` | 분석 진행 상태 (queued/running/done) |
| `POST /analyze` | 분석 트리거 (⚠️ EC2에서는 사용 금지 — 맥에서 수집) |

---

## 3. AWS 리소스 (전부 서울 리전)

| 리소스 | 이름/사양 | 용도 |
|---|---|---|
| EC2 | `billiard-loader` · **t3.small** (2GB+스왑2GB, 디스크 16GB) | 적재 cron + CueCast + FastAPI + 터널 |
| RDS | `billiard-db` · PostgreSQL (프리 티어) | `billiard_turns` 저장. **VPC 내부(172.31.0.0/16)만 5432 허용** |
| S3 | `billiard-turns-mcamp3w` | 결과·export·코드 중계 |
| IAM | `billiard-loader-role` | EC2의 S3 접근 |

EC2 crontab (4줄):
```
*/10 * * * *  S3 → RDS 적재            (~/loader/load_to_db.py)
*/10 * * * *  RDS → export 파일 갱신    (~/app/db/export_db.py)
@reboot       서비스 자동 시작          (~/app/run_services.sh)
0 6 * * *     매일 재시작 = 최신 데이터로 모델 재학습
```

---

## 4. 운영 방법 (치트시트)

### 새 영상 데이터 수집 (맥)
```bash
cd ~/26s-w3-c3-01
echo "https://www.youtube.com/watch?v=XXXX" > jobs/pending/작업명.txt
source db/db.env && venv/bin/python src/process_queue.py
```
→ 이후는 전자동: S3 업로드 → (10분 내) RDS 적재 → (10분 내) export 갱신.

### 새 데이터를 확률 모델에 즉시 반영 (EC2)
```bash
~/app/run_services.sh        # CueCast+FastAPI 재시작 → 최신 export 로 재학습 (수십 초)
curl -s http://127.0.0.1:8765/api/v1/health   # records 숫자로 반영 확인
```
> CueCast는 **시작할 때만** 데이터를 읽는다. 매일 06시 자동 재시작이 있지만,
> 바로 반영하고 싶으면 위 스크립트를 손으로 실행.

### 접속 주소(터널) 확인 (EC2)
```bash
sudo docker logs cuecast-tunnel 2>&1 | grep -o 'https://[^ ]*trycloudflare.com' | head -1
sudo docker logs api-tunnel     2>&1 | grep -o 'https://[^ ]*trycloudflare.com' | head -1
```
> ⚠️ 터널 주소는 **컨테이너가 재시작되면 바뀐다** (EC2 재부팅 포함). 바뀌면 위로 재확인.
> 주소만 알면 누구나 접속 가능하니 팀 내부에만 공유.

### 상태 점검 (EC2)
```bash
tail -5 ~/loader/loader.log            # S3→RDS 적재 로그
tail -3 ~/app/logs_export.log          # export 갱신 로그
wc -l ~/app/billiard_turns_export.jsonl  # 현재 데이터 행 수
tail -30 ~/app/logs_cuecast.log        # CueCast 서버 로그
tail -30 ~/app/logs_fastapi.log        # FastAPI 로그
```

### 서버 코드 업데이트 (EC2)
```bash
cd ~/app && git pull && ~/app/run_services.sh
```

---

## 5. CueCast 웹 사용법 (터널 주소 접속 후)

1. **실시간 분석 탭**에서 상단 입력창에 **유튜브 영상 URL** 붙여넣기
2. 영상이 로드되면:
   - **타임라인 드래그** 또는 `시:분:초` 입력 → `위치 이동`
   - **`이 위치의 리플레이 직전 공 분석`** 클릭 → 서버가 그 위치 이전 12초에서
     마지막 안정 배치를 찾아 **버추얼 당구대 + 성공 확률** 표시
   - URL 연결 시 **자동 분석**도 시작됨 — 재생을 따라가며 공이 멈춘 배치를
     감지할 때마다 확률 자동 갱신 (`자동 분석 중지`로 일시정지)
3. **샷 기록 / 선수 통계** 탭: 축적된 데이터 조회
4. 분석 연산은 전부 **서버(EC2)에서** 실행된다 — 브라우저는 표시만.
   > ⚠️ t3.small(CPU)이라 유튜브 실시간 분석은 느릴 수 있음. 데모 전 미리 확인.

---

## 6. 자주 겪는 문제

| 증상 | 원인/해결 |
|---|---|
| 터널 주소 접속 안 됨 | 주소가 바뀌었을 것 → §4 "접속 주소 확인"으로 새 주소 조회 |
| CueCast `records: 0` | `--shots` 파일 못 찾음 → run_services.sh 로 재시작 (export 파일 경로 포함됨) |
| CueCast 응답 없음(시작 직후) | 시작 시 CatBoost 학습 중 (수십 초) → 잠시 후 재시도 |
| 새 데이터가 확률에 반영 안 됨 | 서버 재시작 필요 → `~/app/run_services.sh` |
| 맥에서 RDS/psql 접속 불가 | 정상 (캠퍼스망 5432 차단) → 데이터 확인은 S3 export 또는 pgweb/터널 |
| 맥에서 `ssh` EC2 접속 불가 | 정상 (캠퍼스망 22 차단) → **EC2 Instance Connect**(브라우저 터미널) 사용 |
| EC2 디스크 부족 | `sudo docker image prune -a -f`, pip `--no-cache-dir`, EBS 확장(무료 30GB까지) |

---

## 7. 데이터 스키마 요약

`billiard_turns` 한 행 = 한 샷. 좌표는 당구대 기준 0~1 정규화 `[x(장축), y(단축)]`, 원점 좌상단.

| 주요 컬럼 | 의미 |
|---|---|
| `video_id`, `turn` | 영상 id + 턴 순번 (복합 PK) |
| `shooter` | 수구 (white/yellow) |
| `before_pos`, `after_pos` | 샷 직전/이후 3구 좌표 (JSONB) |
| `success`, `success_method` | 성공 여부 · 판정 방식(**scoreboard**만 적재) |
| `bank_shot` | 뱅크샷(+2점) 여부 |
| `time_start_s`, `time_end_s` | 영상 내 시간 구간 |

상세 명세·좌표 규약·대칭 변환 매핑은 [DATA_CONTRACT.md](DATA_CONTRACT.md) 참고.
