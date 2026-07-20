# CueCast YOLO 당구공 좌표 분석기

사진에서 흰 공, 노란 공, 빨간 공을 검출하고 당구대 기준 `0~1` 좌표로
변환합니다. 영상에서는 세 공이 일정 시간 모두 정지했을 때만 좌표 이벤트를
JSONL과 CSV로 기록합니다.

## 구성

```text
YOLO/
├─ analyze_video.py             영상 정지 이벤트 분석
├─ analyze_video_auto.py        학습 전 색상 기반 영상 자동 분석
├─ infer_image_auto.py          학습 전 이미지 좌표 자동 추출
├─ infer_image.py               단일 이미지 좌표 추론
├─ train.py                     커스텀 YOLO 학습
├─ config/
│  ├─ billiard_balls.yaml       데이터셋 설정
│  └─ table.example.json        테이블 네 모서리 예시
├─ cuecast_yolo/
│  ├─ detector.py               YOLO 검출 래퍼
│  ├─ geometry.py               원근 좌표 변환
│  ├─ stop_detector.py          정지 상태 머신
│  └─ output.py                 JSONL/CSV 출력
└─ tests/test_core.py
```

## 1. 환경 설치

PyTorch 및 Ultralytics 설치 호환성을 위해 Python 3.11 또는 3.12 가상환경을
권장합니다.

```powershell
cd YOLO
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 데이터셋 준비

각 이미지에서 공을 바운딩 박스로 라벨링하고 다음 구조로 둡니다.

```text
datasets/billiard_balls/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

클래스는 반드시 다음 순서를 사용합니다.

```text
0 white_ball
1 yellow_ball
2 red_ball
```

라벨은 YOLO 형식입니다.

```text
class_id center_x center_y width height
```

모든 좌표와 크기는 이미지 전체를 기준으로 `0~1` 정규화 값입니다.

## 3. 모델 학습

```powershell
python train.py `
  --model weights/yolo11n.pt `
  --data config/billiard_balls.yaml `
  --epochs 100 `
  --imgsz 1280
```

`weights/yolo11n.pt`는 일반 객체로 사전학습된 시작 가중치입니다. 당구공 전용
모델이 아니므로 이미지·영상 추론에 최종 모델로 사용하지 않습니다. 위 학습이
끝나면 생성되는 `best.pt`가 흰 공, 노란 공, 빨간 공 전용 최종 가중치입니다.

학습된 가중치는 기본적으로 다음 위치에 생성됩니다.

```text
runs/billiard_balls/train/weights/best.pt
```

## 4. 테이블 영역 지정

`config/table.example.json`을 복사한 뒤 영상의 플레이 가능한 안쪽 쿠션 영역
네 모서리를 픽셀 좌표로 지정합니다.

```json
{
  "corners": [
    [412, 221],
    [1546, 221],
    [1546, 785],
    [412, 785]
  ]
}
```

순서는 다음과 같습니다.

```text
좌상단 → 우상단 → 우하단 → 좌하단
```

카메라가 고정된 동일 영상 구간에서는 같은 설정을 재사용할 수 있습니다.
카메라가 바뀌는 영상은 구간별 설정이나 테이블 자동 검출 기능이 필요합니다.

## 5. 이미지 좌표 추론

```powershell
python infer_image.py screenshot.png `
  --model runs/billiard_balls/train/weights/best.pt `
  --table config/table.example.json
```

결과:

```text
outputs/image_result.json
outputs/image_annotated.jpg
```

### 학습 모델 없이 이미지 바로 테스트

파란 테이블 전체가 보이는 상단 이미지라면 `best.pt` 없이 바로 테스트할 수
있습니다.

```powershell
python infer_image_auto.py "스크린샷.png"
```

결과는 `outputs/image_auto/coordinates.json`과
`outputs/image_auto/annotated.jpg`에 생성됩니다.

## 6. 영상 정지 좌표 추출

```powershell
python analyze_video.py match.mp4 `
  --model runs/billiard_balls/train/weights/best.pt `
  --table config/table.example.json `
  --sample-fps 10 `
  --save-snapshots
```

결과:

```text
outputs/video/coordinates.jsonl
outputs/video/coordinates.csv
outputs/video/snapshots/
```

### 학습 모델 없이 고정 방송 영상 분석

파란 테이블의 상단 고정 카메라 영상은 색상 기반 검출기로 먼저 분석할 수
있습니다. `--table`을 지정하면 해당 카메라와 다른 사선·클로즈업 장면은
제외합니다.

```powershell
python analyze_video_auto.py video1.mp4 `
  --table config/video1_table.json `
  --output-dir outputs/video1_result `
  --sample-fps 5 `
  --stable-seconds 0.7 `
  --duplicate-threshold 0.025 `
  --min-event-separation 2
```

이 방식은 현재 영상의 초기 데이터 추출 및 YOLO 자동 라벨 생성용입니다.
다른 테이블 색상이나 카메라에는 별도 보정 또는 학습된 `best.pt`를 사용해야
합니다.

JSONL 이벤트 예시:

```json
{
  "event_id": 1,
  "timestamp_seconds": 472.84,
  "frame_number": 14185,
  "positions": {
    "white_ball": [0.097, 0.691],
    "yellow_ball": [0.933, 0.401],
    "red_ball": [0.112, 0.732]
  },
  "timestamp": "00:07:52.840"
}
```

## 정지 감지 옵션

- `--stable-seconds 0.7`: 이 시간 동안 세 공이 안정적이면 정지로 판정
- `--stop-threshold 0.002`: 정지 구간에서 허용할 좌표 흔들림
- `--move-threshold 0.005`: 정지 상태를 해제할 최소 이동 거리
- `--sample-fps 10`: 초당 분석 프레임 수
- `--conf 0.5`: 최소 객체 검출 신뢰도
- `--imgsz 1280`: YOLO 입력 크기

임계값은 모두 테이블 너비·높이를 `0~1`로 정규화한 좌표를 기준으로 합니다.
세 공 중 하나라도 검출되지 않으면 정지 이벤트를 생성하지 않습니다.

## Hybrid 샷 성공률 예측 엔진

Hybrid 엔진은 경기 전 선수 승률을 계산하는 모델이 아니다. 영상에서 검출한
수구와 두 목적구의 포메이션을 입력받아, 해당 배치에서 득점에 성공할 확률을
예측한다.

구현 파일은 `cuecast_yolo/shot_probability.py`이며 로컬 서버에서는
`POST /api/v1/shot-probability` API로 사용한다.

### 전체 처리 흐름

```text
영상에서 흰 공·노란 공·빨간 공 검출
→ 당구대 기준 좌표를 실제 mm 좌표로 변환
→ 선택한 수구를 기준으로 수구·목적구1·목적구2 역할 정규화
→ 연속 좌표 모델 기본 확률 계산
→ 가까운 과거 포메이션의 가중 성공률 계산
→ Adaptive Grid 계층 확률 계산
→ 데이터 양에 따라 세 확률의 가중치 결정
→ 좌표 오차 범위에서 32회 반복 예측
→ 최종 성공률·난이도·신뢰도·경고 플래그 반환
```

### 입력 데이터

API에는 선택한 수구 색상과 샷 직전 세 공 좌표를 전달한다. UI와 검출기는
`0~1` 정규화 좌표를 사용하고, 엔진 내부에서는 실제 당구대 크기에 맞춘 mm
좌표로 변환한다.

```json
{
  "shooter": "white",
  "before": {
    "white": [0.5583, 0.9381],
    "yellow": [0.2336, 0.7072],
    "red": [0.9643, 0.3434]
  },
  "position_error_mm": 25
}
```

`shooter`는 `white` 또는 `yellow`이다. 선택한 공이 수구가 되고, 빨간 공과
나머지 공이 목적구로 정규화된다.

### 1. 연속 좌표 모델

연속 좌표 모델은 현재 포메이션의 기본 성공 확률 `p_model`을 만든다. 학습된
모델 파일이 있으면 CatBoost 모델을 사용하고, 데이터가 적으면 Logistic 모델,
데이터와 학습 모델이 모두 없으면 Bootstrap 기본 확률을 사용한다.

CatBoost 학습 시에는 검출 신뢰도가 낮거나 위치 오차가 큰 레코드의 표본
가중치를 낮춘다.

```text
sampleWeight = detectionConfidence × (1 - positionErrorMm / 100)
```

최솟값은 각각 `0.05`로 제한한다.

연속 모델의 입력 특성은 다음과 같다.

- 수구·목적구1·목적구2의 x, y 좌표
- 수구와 목적구1의 거리
- 수구와 목적구2의 거리
- 두 목적구 사이의 거리
- 수구 기준 두 목적구 사이의 각도
- 세 공이 만드는 삼각형 넓이
- 각 공과 좌·우·상·하 쿠션 사이의 거리
- 각 공의 쿠션 인접 여부
- 각 공 쌍의 밀착 여부
- 수구가 흰 공인지 여부

거리와 좌표는 테이블 크기로 정규화하고, 각도는 `0~1`, 삼각형 넓이는 전체
테이블 넓이 기준 비율로 변환한다. 쿠션 120mm 이내는 쿠션 인접, 공 사이
150mm 이내는 밀착 상태로 처리한다.

### 2. 유사 포메이션 가중 평균

현재 포메이션과 과거 각 포메이션 사이의 거리를 다음과 같이 계산한다.

```text
d = sqrt((d_cue² + d_object1² + d_object2²) / 3)
```

처음에는 30mm 반경 안의 기록을 찾는다. 유효 표본이 부족하면 반경을
15mm씩 확대하며 최대 120mm까지 검색한다.

가까운 과거 기록에는 Gaussian 가중치를 준다.

```text
w_i = exp(-(d_i²) / (2 × 45²))

p_neighbor = sum(w_i × success_i) / sum(w_i)
```

단순 검색 개수 대신 유효 표본 수를 계산한다.

```text
N_eff = sum(w_i)² / sum(w_i²)
```

목표 유효 표본 수는 50이다. 최대 반경까지 유사 기록이 하나도 없으면
`p_neighbor` 대신 `p_model`을 사용한다.

### 3. Adaptive Grid

Adaptive Grid는 당구대의 지역별 성공·실패 통계를 반영한다. 세 공이 속한 셀
조합을 검색 키로 사용한다.

```text
stateKey = (cueCellId, object1CellId, object2CellId)
```

지원하는 해상도는 다음과 같다.

```text
Level 0: 24 × 12
Level 1: 48 × 24
Level 2: 96 × 48
```

전체 테이블을 한 번에 세분화하지 않는다. 현재 배치의 부모 영역이 다음
조건을 만족할 때만 더 세밀한 레벨을 사용한다.

- 부모 영역 표본이 150개 이상
- 관측된 자식 영역의 표본이 각각 25개 이상
- 조건을 만족하는 자식 영역이 2개 이상
- 자식 간 성공률 차이가 15%p 이상이거나 검증 Log Loss가 0.02 이상 개선

학습 데이터 중 샷 ID 해시 기준 약 20%를 그리드 분할 검증용으로 분리한다.
분할 전 부모 확률과 분할 후 자식 확률의 Log Loss를 비교해 실제 개선 여부를
확인한다.

### 4. 계층형 그리드 보정

각 그리드의 원시 성공률을 그대로 사용하지 않는다. 연속 좌표 모델을 최상위
prior로 사용하고, 부모에서 자식으로 내려가면서 점진적으로 보정한다.

```text
p_grid,L = (successes_L + lambda_L × p_parent) /
           (attempts_L + lambda_L)
```

레벨별 prior 강도는 다음과 같다.

```text
Level 0: 30
Level 1: 20
Level 2: 15
```

예를 들어 현재 셀이 1전 1승이고 부모 확률이 72%라면, 원시 성공률 100%를
사용하지 않고 다음처럼 보정한다.

```text
(1 + 15 × 0.72) / (1 + 15) = 73.75%
```

따라서 극소수 성공·실패 기록 때문에 확률이 0% 또는 100%로 급변하지 않는다.

### 5. 최종 Hybrid 확률

최종 성공 확률은 세 구성 요소를 결합한다.

```text
p_final = w_model × p_model
        + w_neighbor × p_neighbor
        + w_grid × p_grid
```

가중치는 유사 포메이션 유효 표본과 현재 그리드 표본 수에 따라 달라진다.

| 데이터 상태 | 연속 모델 | 유사 배치 | Grid |
|---|---:|---:|---:|
| 유효 표본 5 미만 | 85% | 10% | 5% |
| 유효 표본 5~19 | 65% | 25% | 10% |
| 유효 표본 20~79 | 45% | 40% | 15% |
| 유효 표본 80 이상, Grid 100 이상 | 25% | 50% | 25% |
| 유효 표본 80 이상, Grid 100 미만 | 35% | 50% | 15% |

데이터가 적을 때는 연속 좌표 모델이 대부분을 담당하고, 실제 유사 기록과
지역 통계가 쌓일수록 데이터 기반 구성 요소의 영향이 커진다.

### 6. 좌표 오차와 Monte Carlo 예측

YOLO 검출 좌표에는 오차가 있으므로 입력 좌표 한 개만으로 확률을 확정하지
않는다. 각 공의 좌표를 오차 범위 안에서 무작위로 흔든 포메이션 32개를 만들고,
각 포메이션에 전체 Hybrid 계산을 다시 적용한다.

기본 위치 오차는 25mm이다. 쿠션에서 120mm 이내인 공은 최소 35mm 오차를
적용한다. 흔들린 좌표가 테이블을 벗어나면 테이블 경계로 제한한다.

```text
최종 성공률 = 32개 반복 예측의 평균
불확실성 = 32개 예측의 표준편차
난이도 = 1 - 최종 성공률
```

재현 가능한 테스트를 위해 현재 난수 시드는 고정되어 있다.

### 7. 데이터 신뢰도

성공 확률과 데이터 신뢰도는 별도로 계산한다.

```text
sampleScore    = min(N_eff / 80, 1)
gridScore      = min(gridSamples / 100, 1)
positionScore  = max(0, 1 - positionErrorMm / 50)
agreementScore = max(0, 1 - modelDisagreement / 0.30)

confidence = 0.35 × sampleScore
           + 0.20 × gridScore
           + 0.25 × positionScore
           + 0.20 × agreementScore
```

`modelDisagreement`는 연속 모델, 유사 배치, 그리드 확률 중 최댓값과 최솟값의
차이다.

```text
0.75 이상: high
0.45 이상: medium
나머지: low
```

UI에서는 각각 `높음`, `보통`, `낮음`으로 표시한다.

### 8. 진단 플래그

예측 결과에는 다음 진단 플래그가 포함될 수 있다.

- `sparse_neighbors`: 유사 배치 유효 표본이 5 미만
- `model_data_conflict`: 세 구성 요소의 확률 차이가 0.30 이상
- `position_error_too_high`: 유효 위치 오차가 50mm 이상
- `bootstrap_prior`: 학습된 연속 좌표 모델 없이 기본 확률 사용

플래그는 성공 확률을 대체하지 않고, 결과 해석과 데이터 수집 우선순위 결정에
사용한다.

### 출력 예시

```json
{
  "modelVersion": "catboost-coordinate-v1-...",
  "cueBall": "white",
  "successProbability": 0.643,
  "difficulty": 0.357,
  "uncertainty": {
    "standardDeviation": 0.041,
    "positionErrorMm": 25,
    "samples": 32
  },
  "confidence": {
    "score": 0.38,
    "level": "low"
  },
  "components": {
    "modelProbability": 0.67,
    "neighborProbability": 0.52,
    "gridProbability": 0.64,
    "weights": {
      "model": 0.85,
      "neighbor": 0.10,
      "grid": 0.05
    },
    "neighborEffectiveSamples": 2.8,
    "gridSamples": 4,
    "gridLevel": 0
  },
  "flags": ["sparse_neighbors"]
}
```

### 현재 데이터 상태에서의 동작

현재 로컬 서버 실행 기준 샷 기록은 24건이다. Adaptive Grid가 Level 1로
분할되려면 부모 영역에 최소 150개가 필요하므로 현재는 대부분 Level 0을
사용한다. 유사 배치 유효 표본도 적은 경우가 많아 실제 최종 가중치는 주로
다음과 같다.

```text
연속 좌표 모델 85%
유사 포메이션 10%
Adaptive Grid 5%
```

즉 현재 결과는 CatBoost 연속 좌표 모델이 중심이고, 유사 포메이션과 Grid는
보조 역할을 한다. 데이터가 증가하면 코드 변경 없이 유사 포메이션과 Adaptive
Grid의 가중치 및 선택 레벨이 자동으로 증가한다.

## 테스트

YOLO 가중치 없이 좌표 변환과 정지 상태 로직을 테스트할 수 있습니다.

```powershell
python -m unittest discover -s tests -v
```

## 현재 제한 사항

- YouTube URL을 직접 받지 않고 로컬 영상 파일을 입력받습니다.
- 방송 화면 전환과 리플레이를 자동 분류하지 않습니다.
- 카메라가 바뀌면 테이블 모서리 설정도 바꿔야 합니다.
- 실제 검출을 위해 직접 학습한 `best.pt`가 필요합니다.
- Ultralytics를 상용 서비스에 사용할 경우 라이선스를 확인해야 합니다.
