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
