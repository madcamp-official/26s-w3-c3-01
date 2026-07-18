# CueCast 하이브리드 성공 확률 모델

## 처리 흐름

```text
영상
  -> 정지 전/후 배치와 수구, 성공 여부 추출
  -> mm 단위 ShotRecord JSON 생성
  -> 연속 좌표 모델 학습
  -> Gaussian 유사 배치 검색
  -> Adaptive Grid 계층형 보정
  -> 데이터 양에 따른 가중 결합
  -> 좌표 오차 Monte Carlo
  -> Prediction JSON 저장
```

구현은 `cuecast_yolo/shot_probability.py`에 있다. 기본 연속 모델은 외부 의존성이
없는 로지스틱 회귀다. 데이터가 충분해지면 `predict_probability()` 인터페이스를
구현하는 CatBoost/LightGBM 모델로 교체할 수 있다.

현재 `--model-type auto`의 cold-start 정책은 다음과 같다.

- 0건: 설정 가능한 Bootstrap prior
- 1~49건: 정규화 로지스틱 좌표 모델
- 50건 이상이며 성공/실패가 모두 존재: CatBoost
- CatBoost가 설치되지 않았으면 로지스틱 모델 유지

모델을 강제로 선택하려면 `--model-type bootstrap|logistic|catboost`를 사용한다.

## 좌표 및 공 역할 규칙

- 좌표 원점은 탑뷰의 왼쪽 위다.
- x 범위는 `0..2840mm`, y 범위는 `0..1420mm`다.
- `cue`는 실제 수구다.
- `object1`, `object2` 순서는 항상 공 색상 이름의 오름차순으로 고정한다.
- 영상 좌표 `(x, y)`가 0~1이면 `(x*2840, y*1420)`으로 변환한다.

목적구 순서가 데이터마다 바뀌면 같은 배치가 다른 배치로 검색되므로 반드시 위
규칙을 유지해야 한다. 현재 `extract_shot_dataset.py`의 출력은 이 순서를 유지한다.

## 실행

### 현재 DB JSONL 형식

루트의 `data`처럼 한 줄에 한 타구가 저장된 파일은 변환 없이 바로 읽는다.

```json
{
  "video_id": "screenrec2",
  "turn": 1,
  "epoch": 0,
  "shooter": "white",
  "before": {"white": [0.5583, 0.9381], "yellow": [0.2336, 0.7072], "red": [0.9643, 0.3434]},
  "after": {"white": [0.9052, 0.2853], "yellow": [0.5355, 0.9313], "red": [0.9639, 0.343]},
  "success": false
}
```

- 좌표는 0~1이며 엔진에서 x는 2840mm, y는 1420mm로 변환한다.
- `(video_id, turn, epoch)`가 DB의 자연키다.
- 동일한 자연키와 동일한 내용은 한 건으로 제거한다.
- 동일한 자연키인데 내용이 다르면 데이터 충돌 오류를 발생시킨다.
- `shooter`가 수구가 되고 나머지 색상은 고정 순서로 목적구가 된다.
- 파일 안의 JSON이 아닌 구분선은 무시한다.

DB 파일로 바로 모델을 학습한다.

```powershell
python train_probability_model.py ..\data `
  --out outputs/probability_db/model.json
```

데이터가 완전히 없는 초기 배포에서는 다음처럼 prior를 지정할 수 있다.

```powershell
python train_probability_model.py empty.jsonl `
  --model-type bootstrap `
  --bootstrap-probability 0.35 `
  --out outputs/probability_bootstrap/model.json
```

Bootstrap 예측에는 `bootstrap_prior`, 유사 배치가 없으면 `sparse_neighbors` 플래그가
포함된다. 데이터가 들어오면 같은 명령을 다시 실행해 로지스틱 또는 CatBoost 모델로
교체한다.

CatBoost를 명시적으로 학습하려면 다음 명령을 사용한다.

```powershell
python train_probability_model.py ..\data `
  --model-type catboost `
  --iterations 400 `
  --out outputs/probability_catboost/model.json
```

DB에서 조회한 현재 배치 JSON으로 예측한다.

```powershell
python predict_probability.py config/db_shot_state.example.json `
  --shots ..\data `
  --model outputs/probability_db/model.json `
  --out outputs/probability_db/prediction.json
```

응답의 `roles`에는 실제 색상 기준 `cue`, `object1`, `object2` 매핑이 포함된다.

### 기존 변환 스키마

기존 타구 추출 결과를 모델 입력으로 변환한다.

```powershell
python convert_shots_for_probability.py outputs/shot_dataset/shots.json `
  --prefix match_001 `
  --out outputs/probability/shots.json
```

`quality=review`, 성공 여부 없음, 수구 판정 실패 레코드는 기본적으로 제외된다.

연속 좌표 모델을 학습한다.

```powershell
python train_probability_model.py outputs/probability/shots.json `
  --out outputs/probability/model.json
```

현재 배치의 확률을 계산한다. 입력 예시는 `config/shot_state.example.json`이다.

```powershell
python predict_probability.py config/shot_state.example.json `
  --shots outputs/probability/shots.json `
  --model outputs/probability/model.json `
  --out outputs/probability/prediction.json
```

## Prediction JSON 주요 필드

- `successProbability`: 좌표 오차까지 반영한 평균 성공 확률
- `difficulty`: `1 - successProbability`
- `uncertainty.standardDeviation`: 좌표 흔들림에 따른 확률 표준편차
- `uncertainty.perBallPositionErrorMm`: 공별 적용 오차. 쿠션 120mm 이내는 최소 35mm
- `confidence`: 성공 확률과 독립적인 데이터 신뢰도
- `components`: 연속 모델, 이웃, 그리드 값과 각 가중치
- `components.nextGridSplit`: 자식 통계 차이와 검증 Log Loss 개선, 분할 가능 여부
- `flags`: 데이터 희소, 모델 충돌, 좌표 오차 경고

## DB 저장 권장안

원본 재처리와 빠른 검색을 모두 지원하려면 JSON만 단독으로 저장하지 않는다.

### `shots`

- `shot_id` 문자열 PK
- `cue_ball` 문자열
- `before_state` JSON/JSONB
- `after_state` JSON/JSONB
- `success` boolean
- `points` integer
- `player_id`, `player_avg`
- `position_error_mm`, `detection_confidence`
- `layout_vector`: `[cue_x, cue_y, object1_x, object1_y, object2_x, object2_y]`
- `raw_payload` JSON/JSONB

### `predictions`

- `prediction_id` 문자열 PK
- `model_version`
- `shot_id` 또는 현재 영상 이벤트 ID
- `success_probability`, `difficulty`, `confidence_score`
- `payload` JSON/JSONB
- `created_at`

PostgreSQL이면 원본에는 JSONB, 유사 배치 검색에는 `pgvector vector(6)` HNSW
인덱스를 권장한다. 현재 Python 엔진은 메모리에서 정확 검색을 하므로 데이터가 수만
건을 넘으면 DB에서 반경 후보를 먼저 가져온 뒤 Gaussian 가중치를 계산하도록 바꾼다.

## 운영상 주의점

- 학습/예측 데이터 분리는 경기 단위로 한다. 같은 경기의 인접 샷을 양쪽에 넣으면
  검증 점수가 과대평가된다.
- 모델 버전과 그리드 통계 생성 시점을 prediction에 반드시 기록한다.
- `model_data_conflict`가 반복되는 영역은 라벨·수구·좌우 반전을 점검한 뒤 grid
  분할 및 모델 재학습 후보로 등록한다.
- `position_error_too_high`가 있으면 화면에 확률을 확정값처럼 표시하지 않는다.
