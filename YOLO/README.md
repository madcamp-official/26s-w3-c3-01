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
