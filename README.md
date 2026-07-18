# 26s-w3-c3-01
몰입캠프 26s-w3-c3-01 프로젝트 repository

## 당구 영상 → 턴 데이터 자동 추출 파이프라인

유튜브 3쿠션 중계 영상에서 매 턴(샷)의 데이터를 자동으로 뽑는다:

- `before`: 샷 직전 공 3개 위치 (당구대 기준 0~1 정규화, [x, y])
- `shooter`: 수구 (white / yellow) — 정지 배치에서 가장 먼저 움직인 공
- `success`: 3쿠션 성공 여부 — 수구 궤적에서 "두 목적구 접촉 + 두 번째 접촉 전 쿠션 3회"를 직접 계산
- `after`: 샷 이후 공 3개 위치 (`after_source: settled`=정지 확인, `last_seen`=영상 컷으로 마지막 관측값)

### 단일 영상 실행

```bash
venv/bin/python src/extract_turns.py videos/영상.mp4 --outdir results/영상ID --save-frames
```

출력: `results/영상ID/turns.jsonl`, `turns.csv`, (옵션) `qa/` 턴별 검증용 프레임.

### 유튜브 링크 큐 자동화

```bash
# 작업 추가 (한 줄에 링크 1개)
echo "https://www.youtube.com/watch?v=XXXX" > jobs/pending/작업이름.txt

# 워커 1회 실행 (다운로드 → 추출 → results/에 저장)
venv/bin/python src/process_queue.py
```

crontab 등록 (10분마다 큐 확인, 이중 실행은 락으로 방지):

```
*/10 * * * * cd /Users/parkminsu/26s-w3-c3-01 && venv/bin/python src/process_queue.py >> logs/worker.log 2>&1
```

필요 도구: `venv/bin/pip install yt-dlp` + JS 런타임(node)이 PATH에 있어야 한다.

### 파일 구성

엔진 코드는 `src/` 에, 실행·유틸(DB·배포)은 각 폴더에 있다.

| 파일 (`src/`) | 역할 |
|---|---|
| `detect_pipeline.py` | 당구대 꼭짓점 검출 + 픽셀→정규화 좌표 변환 |
| `detect_video.py` | 프레임별 탐지/미니맵 합성 영상 + 시뮬레이션 확률 (시각화용) |
| `extract_turns.py` | 턴 단위 데이터 추출 (배치/서버용 핵심 모듈) |
| `process_queue.py` | 유튜브 링크 큐 워커 (crontab용) |
| `simulate.py` | 3쿠션 물리 시뮬레이션 (성공 확률 추정) |
| `best_3cls.pt` | YOLO 공 탐지 모델 (white/yellow/red) |

### turns.jsonl 스키마 예시

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
