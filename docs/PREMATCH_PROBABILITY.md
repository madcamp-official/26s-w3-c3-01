# 경기 전 승률 예측

`final_dataset_2026_start.zip`의 2026 시즌 시작 스냅샷을 PostgreSQL에 적재하고,
CueCast의 `경기 전 승률 예측` 화면에서 선수별 예상 승률을 조회한다. 원본 ZIP과 선수
이미지는 저장소에 커밋하지 않는다.

## 계산식

최종 승률은 아래 다섯 항목의 선수 A 승리 확률을 데이터 신뢰도로 재가중한 선형 결합이다.

| 항목 | 기본 가중치 | 내용 |
|---|---:|---|
| Elo | 30% | Elo 차이를 로지스틱 확률로 변환 |
| 통산 승률 | 17.5% | 50% 의사 표본을 적용한 보정 승률 비교 |
| 현재 시즌 | 5% | 2026 시즌 경기 수에 따른 신뢰도 적용 |
| 최근 흐름 | 2.5% | 최근 5경기 40% + 최근 10경기 60% |
| 세부 경기력 | 45% | AVG·TS·BRS·5HS·HR 표준화 점수 비교 |

각 항목의 실효 가중치는 `기본 가중치 × 신뢰도`이며, 합이 1이 되도록 다시 정규화한다.
상대 전적은 화면의 참고 정보로만 표시하고 최종 승률에는 포함하지 않는다.

## DB 적재

의존성을 설치한 뒤 현재 서비스와 같은 `DATABASE_URL`을 사용한다.

```bash
python -m pip install -r YOLO/requirements.txt
python db/import_prematch_dataset.py /path/to/final_dataset_2026_start.zip
```

적재 스크립트는 [prematch_schema.sql](../db/prematch_schema.sql)을 먼저 적용한 뒤 다음
테이블을 upsert한다.

- `prematch_players`: 선수 기본 정보와 UI용 이미지
- `prematch_player_features`: 시점별 Elo·승패·최근 흐름·경기력 스냅샷
- `prematch_head_to_head`: 상대 전적 참고 정보
- `prematch_league_metric_baselines`: 지표 표준화 기준

이미지를 DB에 넣지 않으려면 `--skip-images`를 사용한다. 이 경우 UI는 선수 이름의 첫
글자를 아바타로 표시한다.

## 로컬 실행

DB 없이 전달받은 데이터셋을 검증할 때만 압축을 푼 디렉터리를 지정할 수 있다.

```powershell
$env:CUECAST_PREMATCH_DATASET_ROOT = "C:\data\final_dataset_2026_start"
python YOLO/local_probability_server.py
```

`DATABASE_URL`이 있으면 PostgreSQL을 우선 사용한다. 둘 다 없으면 기존 샷 성공률 API는
정상 동작하지만 경기 전 승률 화면에는 연결 안내가 표시된다.

## API

- `GET /api/v1/prematch/players?league=PBA&active_only=true`: 선택 가능한 선수 목록
- `POST /api/v1/match-probability`: 선수 코드 기반 경기 전 승률
- `GET /api/v1/players/{player_code}/image?league=PBA`: 선수 이미지

예측 요청 예시:

```json
{
  "league": "PBA",
  "season_code": 2026,
  "player_a_code": "M0017784",
  "player_b_code": "M0017160"
}
```
