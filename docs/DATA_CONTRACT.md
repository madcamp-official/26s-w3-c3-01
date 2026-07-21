# 데이터 계약 — 당구 턴 데이터 (확률 엔진 입력용)

> 데이터 파이프라인(영상 → 추출 → RDS) 담당이 확률 엔진(CatBoost/Neighbor/Grid) 담당에게
> 넘기는 명세. **좌표 규약(§3)이 대칭 변환의 전제**이니 반드시 먼저 확인할 것.

---

## 1. 데이터가 무엇인가

- 1소스: PBA/LPBA 3쿠션 중계 영상 → YOLO로 공 검출 → 매 "턴(샷)"의 데이터.
- 저장 위치: **RDS PostgreSQL `billiard` DB → `billiard_turns` 테이블**.
- 한 행 = 한 샷. `before`(샷 직전 3구 위치), `shooter`(수구), `success`(성공 여부), `after`(샷 이후 위치).
- 현재 규모: **약 503턴** (영상이 추가될수록 계속 증가).

---

## 2. DB 접속

| 항목 | 값 |
|---|---|
| Host | `billiard-db.cx0e6meogl9d.ap-northeast-2.rds.amazonaws.com` |
| Port | `5432` |
| Database | `billiard` |
| User | `postgres` |
| Password | **문서·git에 넣지 않음** — 별도로 안전하게 전달 (EC2 `~/loader/loader.env` 에도 있음) |

**⚠️ 접근 제약 (중요):** RDS는 **VPC 내부에서만** 접속됩니다.
- ✅ **같은 VPC(`vpc-0c7029f409228a974`)의 EC2**에서 접속 → 됨 (SG가 `172.31.0.0/16` 허용).
- ❌ 개인 노트북·캠퍼스망에서 직접 접속 → **안 됨** (기관망이 5432 차단).
- → **확률 엔진은 같은 VPC 안의 EC2에서 실행**해야 DB에 붙습니다. 밖에서 개발·테스트하려면
  아래 §6의 S3 export(`billiard_turns_export.jsonl`)를 파일로 받아 쓰세요.

---

## 3. 좌표 규약 ⭐ (대칭 변환의 전제)

모든 위치는 **당구대 경기면 기준 0~1 정규화** 값 `[x, y]`.

```
        x: 0 ──────────────(장축, 2.84m)────────────▶ 1
   y:0  ┌───────────────────────────────────────────┐
    │   │ (0,0)좌상                          우상(1,0) │
 (단축, │                  당구대                      │
 1.42m)│                                             │
    ▼   │ (0,1)좌하                          우하(1,1) │
   y:1  └───────────────────────────────────────────┘
```

- **x축 = 장축**(긴 변, 실제 2.84m), 왼→오른쪽으로 0→1.
- **y축 = 단축**(짧은 변, 실제 1.42m), 위→아래로 0→1.
- 원점 `(0,0)` = **좌상단** 꼭짓점. `(1,1)` = 우하단.
- 실좌표가 필요하면: `x_m = x * 2.84`, `y_m = y * 1.42`.

### 이 좌표계에서의 대칭 연산 (엔진의 8변환과 1:1 매핑)

| 변환 | 좌표 연산 |
|---|---|
| 원본 | `(x, y)` 그대로 |
| **좌우 대칭** (장축 기준 반전) | `x → 1 - x`, `y` 유지 |
| **상하 대칭** (단축 기준 반전) | `y → 1 - y`, `x` 유지 |
| 좌우+상하 | `x → 1-x`, `y → 1-y` |
| **O1·O2 교환** | 두 **목적구**의 좌표를 서로 swap (수구는 그대로) |

→ 8변환 = {원본, 좌우, 상하, 좌우+상하} × {O1O2 유지, 교환}. 이 규약대로 변환해야
`symmetric_layouts` / `canonical_grid_key` / `symmetric_layout_distance`가 데이터와 정확히 맞습니다.

---

## 4. 공·수구 의미 (O1·O2 교환에 필요)

- 공은 3개: `white`, `yellow`, `red`.
- **수구(cue) = `shooter`** — 항상 `white` 또는 `yellow` (3쿠션에서 각 선수의 수구). `red`는 수구가 안 됨.
- **목적구 2개(O1·O2)** = 수구가 아닌 나머지 두 공:
  - `shooter="white"` → 목적구 = `{yellow, red}`
  - `shooter="yellow"` → 목적구 = `{white, red}`
- "O1·O2 교환" 대칭 = 이 두 목적구의 좌표를 swap.

---

## 5. 테이블 스키마 (`billiard_turns`)

| 컬럼 | 타입 | 의미 |
|---|---|---|
| `video_id` | text | 유튜브 영상 id |
| `turn` | int | 영상 내 턴 순번 (1부터). `(video_id, turn)` = 복합 PK |
| `epoch` | int | 연속 구간(클립) 번호. **값이 튀면 카메라 컷 경계** (앞뒤 턴이 불연속) |
| `shooter` | text | 수구: `white` \| `yellow` (점수판 기준) |
| `success` | bool | 3쿠션 성공 여부. 점수판 판정이라 **거의 항상 true/false** (NULL은 폴백 케이스만) |
| `success_method` | text | 현재는 **`scoreboard`**. (`trajectory`/`insufficient`는 구버전 레거시) |
| `bank_shot` | bool | **뱅크샷(+2점) 여부** — 점수판 판정에서만 채워짐 |
| `coverage` | real | 샷 구간 탑뷰 관측 비율(0~1). **낮으면 좌표가 근사치**(라벨은 여전히 점수판 근거라 유효) |
| `cushions_before_2nd` | int | (레거시 궤적 판정용) 점수판 턴에서는 `NULL` |
| `hits` | jsonb | (레거시) 점수판 턴에서는 `[]` |
| `before_pos` | jsonb | **샷 직전** 3구 좌표 `{"white":[x,y],"yellow":[x,y],"red":[x,y]}` |
| `after_pos` | jsonb | 샷 이후 3구 좌표 (동일 구조). 턴 N의 `after` = 턴 N+1의 `before` (연결됨) |
| `after_source` | text | 좌표 출처: `still`/`still_near`(정지 배치·정확) · `obs_near`/`obs_far`(공별 관측·근사) |
| `frame_start`,`frame_end` | int | 원본 영상 프레임 구간 (= 이닝 이벤트 경계) |
| `time_start_s`,`time_end_s` | real | 초 단위 구간 |
| `loaded_at` | timestamptz | DB 적재 시각 |

> `bank_shot`, `total_delta`, `totals` 등 상세 판정 근거는 원본 `turns.jsonl`의 `success_detail`에도
> 들어 있다 (DB 컬럼은 위 표가 전부). `bank_shot`은 DB 컬럼으로도 승격돼 있음.

### 데이터 주의사항
- **라벨(수구·성공·뱅크)은 방송 점수판 근거**라 신뢰도 높음. `success`가 NULL인 행은 사실상 없음.
- **좌표 품질은 `coverage`/`after_source`로 판단**: `coverage`가 낮거나 `after_source`가 `obs_far`면
  그 턴의 공 위치는 근사치(경계 부근에 정지 배치가 없어 움직이는 중 관측을 쓴 경우). 위치 정밀도가
  중요한 학습이라면 `coverage >= 0.3` 등으로 거르는 걸 권장.
- `epoch`로 클립 경계를 알 수 있음 — 점수판 주도 턴은 대부분 `epoch=0`.

---

## 6. 바로 쓰는 예시

### SQL — 학습 데이터 조회 (VPC 안 EC2에서)
```sql
SELECT shooter, before_pos, after_pos, success, bank_shot, coverage
FROM billiard_turns
WHERE success IS NOT NULL
  AND coverage >= 0.3          -- 좌표가 근사치인 저품질 턴 제외 (선택)
ORDER BY video_id, turn;
```

### Python (psycopg2) — 로드
```python
import os, psycopg2
conn = psycopg2.connect(os.environ["DATABASE_URL"])   # loader.env 의 값 사용
cur = conn.cursor()
cur.execute("""SELECT shooter, before_pos, success
               FROM billiard_turns WHERE success IS NOT NULL""")
for shooter, before, success in cur:
    # before 는 psycopg2 가 jsonb → dict 로 파싱: {"white":[x,y],"yellow":[..],"red":[..]}
    cue = before[shooter]                     # 수구 좌표
    objs = [before[b] for b in ("white","yellow","red") if b != shooter]  # 목적구 2개
    # → 여기서 대칭 변환(§3) 적용 후 특징 생성
```

### VPC 밖에서 개발할 때 — S3 export 파일 사용
DB 대신 파일로 받으려면(캠퍼스망 OK): EC2에서 `db/export_db.py` 실행 → `s3://billiard-turns-mcamp3w/exports/billiard_turns.jsonl` 생성 → 노트북에서 그 파일 다운로드.
JSONL 각 줄이 테이블 한 행과 동일한 필드 구성입니다.

---

## 7. 샘플 레코드
```json
{"video_id": "당구분석3", "turn": 2, "epoch": 0, "shooter": "yellow",
 "before_pos": {"white": [0.66, 0.24], "yellow": [0.68, 0.29], "red": [0.11, 0.13]},
 "after_pos":  {"white": [0.62, 0.44], "yellow": [0.34, 0.41], "red": [0.11, 0.94]},
 "success": true, "success_method": "scoreboard", "bank_shot": true,
 "coverage": 0.59, "cushions_before_2nd": null, "hits": [],
 "after_source": "obs_near", "time_start_s": 4.9}
```
> 뱅크샷 예시: 이닝 원형 1→3 **그리고** 총점 박스 +2가 함께 확인돼 `bank_shot=true`.

---

## 8. 역할 경계
- **데이터 담당(파이프라인)**: 위 스키마·좌표 규약대로 `billiard_turns` 에 데이터를 계속·정확히 적재. 스키마 변경 시 이 문서 갱신.
- **엔진 담당**: `billiard_turns` 를 읽어 CatBoost/Neighbor/Grid 실행. 좌표 규약(§3)·수구 의미(§4)를 그대로 사용.
- 스키마/좌표 규약을 바꿔야 하면 **양쪽 합의 후 이 문서부터 수정**.
