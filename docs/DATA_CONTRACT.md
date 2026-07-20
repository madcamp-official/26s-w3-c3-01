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
| `shooter` | text | 수구: `white` \| `yellow` |
| `success` | bool | 3쿠션 성공 여부. **`NULL` = 판정 보류** (궤적 관측 부족) → 학습 시 제외 권장 |
| `success_method` | text | `trajectory` \| `insufficient` |
| `coverage` | real | 샷 구간 수구 관측 비율(0~1). 낮으면 판정 신뢰도 낮음 |
| `cushions_before_2nd` | int | 2번째 목적구 접촉 전 쿠션 수 (성공 근거) |
| `hits` | jsonb | 접촉 순서 예: `["red","white"]` |
| `before_pos` | jsonb | **샷 직전** 3구 좌표 `{"white":[x,y],"yellow":[x,y],"red":[x,y]}` |
| `after_pos` | jsonb | 샷 이후 3구 좌표 (동일 구조) |
| `after_source` | text | `settled`=정지 확인 / `last_seen`=영상 컷으로 마지막 관측(근사값) |
| `frame_start`,`frame_end` | int | 원본 영상 프레임 구간 |
| `time_start_s`,`time_end_s` | real | 초 단위 구간 |
| `loaded_at` | timestamptz | DB 적재 시각 |

### 데이터 주의사항
- **학습엔 `success IS NOT NULL`만 사용** (NULL은 판정 불가 케이스).
- `after_source='last_seen'` 은 공이 완전히 멈추기 전 값이라 `after_pos`가 근사입니다. `before_pos`는 항상 정지 상태라 신뢰 가능.
- `epoch`로 클립 경계를 알 수 있음 — "다음 턴 수구 연속성" 같은 시퀀스 가정은 같은 `epoch` 안에서만 유효.

---

## 6. 바로 쓰는 예시

### SQL — 학습 데이터 조회 (VPC 안 EC2에서)
```sql
SELECT shooter, before_pos, success, cushions_before_2nd, coverage
FROM billiard_turns
WHERE success IS NOT NULL
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
{"video_id": "XFE136kVJ-c", "turn": 1, "epoch": 0, "shooter": "white",
 "before_pos": {"white": [0.50, 0.39], "yellow": [0.74, 0.50], "red": [0.26, 0.50]},
 "after_pos":  {"white": [0.14, 0.12], "yellow": [0.48, 0.19], "red": [0.27, 0.13]},
 "success": true, "success_method": "trajectory", "coverage": 1.0,
 "cushions_before_2nd": 4, "hits": ["red", "white"],
 "after_source": "settled", "time_start_s": 27.0}
```

---

## 8. 역할 경계
- **데이터 담당(파이프라인)**: 위 스키마·좌표 규약대로 `billiard_turns` 에 데이터를 계속·정확히 적재. 스키마 변경 시 이 문서 갱신.
- **엔진 담당**: `billiard_turns` 를 읽어 CatBoost/Neighbor/Grid 실행. 좌표 규약(§3)·수구 의미(§4)를 그대로 사용.
- 스키마/좌표 규약을 바꿔야 하면 **양쪽 합의 후 이 문서부터 수정**.
