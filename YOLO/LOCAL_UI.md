# CueCast 로컬 UI

KAIST VM 없이 UI와 확률 엔진을 한 PC에서 실행할 수 있다.

```powershell
cd YOLO
.\.venv\Scripts\python.exe local_probability_server.py
```

브라우저에서 `http://127.0.0.1:8765`를 연다. 서버는 기본적으로 저장소 루트의
`data` 파일과 `outputs/probability_db_catboost/model.json`을 읽는다. 모델 파일이
없으면 현재 데이터 양에 맞춰 Bootstrap, Logistic 또는 CatBoost 모델을 선택한다.

## 탐지기 연결

탐지기는 정규화된 세 공 좌표를 다음 API로 보내면 된다.

```http
POST /api/v1/detection
Content-Type: application/json

{
  "shooter": "white",
  "before": {
    "white": [0.18, 0.30],
    "yellow": [0.50, 0.70],
    "red": [0.82, 0.42]
  },
  "position_error_mm": 25
}
```

서버가 즉시 하이브리드 엔진으로 확률을 계산해 최신 탐지 상태에 저장한다. UI는
`GET /api/v1/detection/latest`를 주기적으로 확인하여 버추얼 당구대와 성공률을
함께 갱신한다.

YouTube URL을 연결하면 타임라인 또는 `시:분:초` 입력으로 분석 위치를 고를 수
있다. `이 위치의 리플레이 직전 공 분석`을 누르면 서버가 선택 위치 이전 12초를
탐색하고, 기존 상단 테이블 공 검출기와 컷 직전 버퍼로 마지막 안정 배치를 찾는다.
검출된 세 공은 버추얼 당구대와 확률 엔진에 동시에 반영된다.

영상 연결 시 자동 분석도 함께 시작된다. 서버가 재생 속도에 맞춰 스트림을 계속
읽고 공이 정지한 배치 또는 리플레이 컷 직전 배치를 찾을 때마다 UI와 확률을 자동
갱신한다. 타임라인 커서를 드래그하거나 시간을 입력해 `위치 이동`을 누르면 영상과
분석 워커가 같은 위치에서 다시 시작한다. `자동 분석 중지` 버튼은 테스트 중 분석만
잠시 멈출 때 사용한다.

브라우저 보안 때문에 iframe 픽셀을 직접 읽는 방식은 사용하지 않는다. 서버가
`yt-dlp`로 같은 YouTube 영상의 분석용 스트림을 열고 OpenCV 검출을 수행한다.
현재 기본 테이블 보정은 `config/video1_table.json`이며 같은 PBA 방송 화면을
기준으로 한다. 다른 카메라 구성에는 별도 테이블 보정 파일이 필요할 수 있다.

## 주요 API

- `GET /api/v1/health`: 데이터 수와 모델 버전 확인
- `POST /api/v1/shot-probability`: 좌표 한 건의 확률 직접 계산
- `POST /api/v1/detection`: 탐지 좌표 저장 및 확률 계산
- `GET /api/v1/detection/latest`: UI가 읽는 최신 포메이션과 예측 조회
- `POST /api/v1/youtube/info`: YouTube 영상 제목과 길이 조회
- `POST /api/v1/youtube/analyze`: 선택 위치 이전의 컷 직전 배치 검출 및 확률 계산
- `POST /api/v1/youtube/live/start`: 현재 재생 위치부터 연속 자동 분석 시작
- `POST /api/v1/youtube/live/stop`: 연속 자동 분석 중지
- `GET /api/v1/youtube/live/status`: 워커 위치와 상태 조회
