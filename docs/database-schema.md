# CueCast DB 스키마 문서

## 1. 문서 개요

CueCast의 PostgreSQL 데이터는 목적에 따라 세 영역으로 나뉩니다.

1. **영상 샷 데이터:** `public.billiard_turns`, `public.billiard_ingest_log`
2. **2026 선수 원본·운영 데이터:** `cuecast.*`
3. **경기 전 승률 서비스용 투영 데이터:** `prematch_*`

`cuecast.*`는 원본 CSV 구조와 시즌 갱신을 보존하는 상세 데이터 계층이고, `prematch_*`는 경기 전 확률 API가 필요한 값을 빠르게 조회하도록 정리한 서비스 계층입니다. 두 계층을 같은 의미의 중복 원본으로 취급하지 않습니다.

현재 통합 서비스는 로컬 CueCast 서버가 EC2 SSH 터널을 통해 RDS에 접근하는 방식으로 검증했습니다. 배포 환경에서 YouTube URL 분석과 DB 연동을 동시에 실행할 때 접근 거부가 발생했으므로, RDS를 인터넷에 직접 공개하는 방식은 사용하지 않습니다.

---

## 2. 전체 관계

```mermaid
erDiagram
  BILLIARD_INGEST_LOG ||--o{ BILLIARD_TURNS : records

  PLAYER_MASTER ||--o{ PLAYER_ALIASES : has
  PLAYER_MASTER ||--|| PLAYER_RUNTIME_STATE : starts_with
  PLAYER_MASTER ||--|| SEASON_PLAYER_STATE : accumulates
  PLAYER_MASTER ||--o{ RECENT_MATCH_EVENTS_SEED : has
  PLAYER_MASTER ||--o{ PLAYER_MATCH_DETAIL_STATS : records
  MATCHES ||--o{ PLAYER_MATCH_DETAIL_STATS : contains

  PREMATCH_PLAYERS ||--o{ PREMATCH_PLAYER_FEATURES : has
  PREMATCH_PLAYERS ||--o{ PREMATCH_HEAD_TO_HEAD : references
  PREMATCH_LEAGUE_METRIC_BASELINES }o--|| PREMATCH_PLAYER_FEATURES : normalizes
```

### 2.1 데이터 흐름

```text
영상 → turns.jsonl → S3 → public.billiard_turns → export JSONL → 샷 성공률 모델

final_dataset_2026_start CSV
  ├─ db/load_players_dataset.py → cuecast.* 상세 테이블
  └─ db/import_prematch_dataset.py → prematch_* 서비스 테이블
                                      ↓
                              경기 전 승률 API
```

---

## 3. `public` 영상 샷 데이터

### 3.1 `public.billiard_turns`

한 행은 한 영상의 한 턴 또는 샷을 의미합니다.

| 컬럼 | 타입 | Null | 설명 |
|---|---|---:|---|
| `video_id` | `TEXT` | N | YouTube 영상 식별자 |
| `turn` | `INT` | N | 영상 안의 턴 순번 |
| `epoch` | `INT` | Y | 탑뷰 연속 구간 번호 |
| `shooter` | `TEXT` | Y | 수구 색상, 서비스 기준 `white` 또는 `yellow` |
| `success` | `BOOLEAN` | Y | 3쿠션 득점 성공 여부 |
| `success_method` | `TEXT` | Y | 적재 데이터 기준 `scoreboard` |
| `coverage` | `REAL` | Y | 샷 구간 수구 관측 비율 |
| `cushions_before_2nd` | `INT` | Y | 두 번째 목적구 전 쿠션 수, QA용 |
| `bank_shot` | `BOOLEAN` | Y | 점수 +2 뱅크샷 여부 |
| `hits` | `JSONB` | Y | 접촉 순서 추정, QA용 |
| `before_pos` | `JSONB` | N | 샷 직전 세 공 좌표 |
| `after_pos` | `JSONB` | N | 샷 이후 세 공 좌표 |
| `after_source` | `TEXT` | Y | 좌표 출처 |
| `frame_start` | `INT` | Y | 시작 프레임 |
| `frame_end` | `INT` | Y | 종료 프레임 |
| `time_start_s` | `REAL` | Y | 영상 시작 시간, 초 |
| `time_end_s` | `REAL` | Y | 영상 종료 시간, 초 |
| `loaded_at` | `TIMESTAMPTZ` | N | DB 적재 시간 |

**Primary Key:** `(video_id, turn)`

**Indexes:**

- `idx_billiard_turns_video(video_id)`
- `idx_billiard_turns_shooter(shooter)`
- `idx_billiard_turns_success(success)`

#### 좌표 JSON 예시

```json
{
  "white": [0.66, 0.24],
  "yellow": [0.68, 0.29],
  "red": [0.11, 0.13]
}
```

#### 무결성 규칙

- `before_pos`, `after_pos`에는 세 공이 모두 있어야 합니다.
- 좌표는 당구대 좌상단을 원점으로 하는 `0~1` 정규화 값입니다.
- 범위 밖 좌표는 추출 단계 또는 로더에서 제외합니다.
- 학습 및 운영 적재는 점수판으로 성공 여부를 판정한 턴만 사용합니다.
- `schema.sql`의 오래된 주석이나 허용 값보다 `load_to_db.py`와 `EXTRACTION_CRITERIA.md`의 scoreboard-only 정책을 우선합니다.

### 3.2 `public.billiard_ingest_log`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `video_id` | `TEXT PK` | 적재한 영상 식별자 |
| `n_turns` | `INT` | 적재한 턴 수 |
| `loaded_at` | `TIMESTAMPTZ` | 마지막 적재 시간 |

영상 단위 적재 완료 여부와 결과 행 수를 확인하는 운영 로그입니다.

---

## 4. `cuecast` 선수 상세 데이터

### 4.1 `cuecast.player_master`

선수 식별자, 이름, 리그와 이미지 메타데이터를 저장합니다.

| 주요 컬럼 | 설명 |
|---|---|
| `player_code` | 선수 고유 코드, PK |
| `league` | `PBA` 또는 `LPBA` |
| `player_name`, `player_name_short` | 정식 이름과 UI 축약 이름 |
| `active_2026_roster` | 2026 시즌 활성 명단 여부 |
| `has_prior_career_stats` | 과거 통산 데이터 존재 여부 |
| `image_file`, `image_url` | 로컬 상대경로 또는 원본 URL |
| `image_is_placeholder` | 실제 선수 이미지 여부 판별 |
| `master_source` | 마스터 생성 출처 |

### 4.2 `cuecast.player_aliases`

OCR 이름이나 다양한 표기를 선수 코드에 연결합니다.

| 키 | 설명 |
|---|---|
| PK | `(player_code, alias)` |
| `normalized_alias` | 공백·기호·대소문자를 정리한 비교용 이름 |
| `alias_source` | 별칭 생성 출처 |

### 4.3 `cuecast.player_runtime_state`

2026 시즌 시작 직전의 선수별 고정 스냅샷입니다.

주요 데이터 묶음:
