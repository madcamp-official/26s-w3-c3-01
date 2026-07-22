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
- `POST /api/v1/match-probability`: 두 선수 AVG와 세트 수 기반 경기 전 승리 확률
- `POST /api/v1/shot-probability`: 좌표 한 건의 확률 직접 계산
- `POST /api/v1/detection`: 탐지 좌표 저장 및 확률 계산
- `GET /api/v1/detection/latest`: UI가 읽는 최신 포메이션과 예측 조회
- `POST /api/v1/youtube/info`: YouTube 영상 제목과 길이 조회
- `POST /api/v1/youtube/analyze`: 선택 위치 이전의 컷 직전 배치 검출 및 확률 계산
- `POST /api/v1/youtube/live/start`: 현재 재생 위치부터 연속 자동 분석 시작
- `POST /api/v1/youtube/live/stop`: 연속 자동 분석 중지
- `GET /api/v1/youtube/live/status`: 워커 위치와 상태 조회

## Chrome 확장 프로그램

1. 로컬 서버를 실행한다.
2. Chrome에서 `chrome://extensions`를 열고 개발자 모드를 켠다.
3. `압축해제된 확장 프로그램을 로드합니다`에서 `YOLO/extension`을 선택한다.
4. YouTube 영상 페이지를 열고 CueCast 아이콘을 누르면 Side Panel과 자동 분석이 시작된다.

UI만 빠르게 수정할 때는 `http://127.0.0.1:8765/extension-preview`에서 확인한다.
# 실시간 경기 승률 데이터 연결

실시간 승률은 별도 더미 값을 사용하지 않고 경기 전 승률 기능의
`PrematchService`를 그대로 호출한다. 점수판 OCR 이름(또는 사용자가 수정한 이름)을
PBA/LPBA 선수 코드로 바꾼 뒤 다음 값을 실시간 DP 입력으로 사용한다.

- 경기 전 전체 경기 승률: `playerA.winProbability`
- 선수 AVG: `playerA.metrics.AVG`, `playerB.metrics.AVG`
- 데이터 출처와 모델 버전: `dataSource`, `modelVersion`

PostgreSQL을 사용할 때는 원격 `main`의 경기 전 승률 기능과 동일하게 설정한다.

```powershell
$env:DATABASE_URL = "postgresql://user:password@host:5432/cuecast"
python local_probability_server.py
```

로컬 검증 데이터셋을 사용할 때는 `CUECAST_PREMATCH_DATASET_ROOT`를 설정한다.
두 환경변수가 모두 없거나 선수의 `AVG`가 없으면 임의 값으로 계산하지 않고 UI에
연결 오류를 표시한다.

RDS가 VPC 내부 접속만 허용하는 로컬 개발 환경에서는 SSH 터널과 서버를 함께 실행한다.
`.env`의 `DATABASE_URL` 호스트는 `127.0.0.1:15432`를 사용한다.

```powershell
cd YOLO
.\start_with_ssh_tunnel.ps1
```

추가 API:

- `GET /api/v1/live-match-probability/latest`: 최신 실시간 경기 승률 조회
- `POST /api/v1/live-match/players`: OCR 이름 대신 사용할 선수 이름 설정
