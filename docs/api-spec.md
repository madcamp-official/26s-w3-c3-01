# CueCast API 명세서

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| 기본 URL | `http://127.0.0.1:8765` |
| 프로토콜 | HTTP |
| 데이터 형식 | JSON |
| 문자 인코딩 | UTF-8 |
| 인증 | 현재 로컬 데모 API는 별도 인증 없음 |
| CORS | `Access-Control-Allow-Origin: *` |
| 캐시 | JSON 상태 API는 `no-store` |
| 서버 | Python `ThreadingHTTPServer` |
| 공식 검증 환경 | 로컬 서버 `127.0.0.1:8765` |
| 시스템 의존성 | Tesseract 실행 프로그램, FFmpeg, yt-dlp |
| 원격 배포 상태 | YouTube URL 분석과 DB 연동 동시 실행 시 접근 거부가 발생해 전체 기능은 아직 로컬 전용 |

모든 POST 요청은 JSON object body를 사용합니다. 빈 입력인 정지·초기화 요청도 현재 서버 구현상 `{}` body를 전송해야 합니다.

---

## 2. 공통 응답

### 2.1 성공

HTTP `200`, 생성·저장은 `201`, 비동기 시작·동기화는 `202`를 사용합니다.

### 2.2 오류

```json
{
  "error": "invalid_request",
  "detail": "shooter는 white 또는 yellow여야 합니다"
}
```

| HTTP | `error` 예시 | 의미 |
|---:|---|---|
| 400 | `invalid_request` | 누락 필드, 잘못된 JSON, 허용되지 않은 값 |
| 404 | `not_found`, `image_not_found` | endpoint 또는 이미지 없음 |
| 422 | `prematch_prediction_failed`, `analysis_failed` | 입력은 읽었지만 예측·분석 수행 실패, YouTube 접근 거부 또는 OCR 실행 실패 포함 |
| 503 | `prematch_data_unavailable` | 경기 전 DB 또는 데이터셋 사용 불가 |

---

## 3. Endpoint 요약

### GET

| Endpoint | 설명 |
|---|---|
| `/api/v1/health` | 서버, 샷 모델과 경기 전 데이터 상태 |
| `/api/v1/prematch/players` | 선수 목록 조회 |
| `/api/v1/players/{player_code}/image` | 선수 이미지 |
| `/api/v1/detection/latest` | 최신 공 배치·샷 확률·점수판 |
| `/api/v1/youtube/live/status` | 실시간 분석 워커 상태 |
| `/api/v1/live-match-probability/latest` | 최신 실시간 세트 승률 |

### POST

| Endpoint | 설명 |
|---|---|
| `/api/v1/shot-probability` | 세 공 좌표 기반 샷 성공률 |
| `/api/v1/detection` | 외부 검출 결과 저장 및 확률 계산 |
| `/api/v1/match-probability` | 경기 전 선수 승률 |
| `/api/v1/youtube/info` | YouTube 영상 정보 |
| `/api/v1/youtube/analyze` | 특정 시점 이전 안정 배치 분석 |
| `/api/v1/youtube/live/start` | 실시간 분석 시작 |
| `/api/v1/youtube/live/stop` | 실시간 분석 정지 |
| `/api/v1/youtube/live/sync` | 재생 시점 동기화 |
| `/api/v1/youtube/live/shooter` | 수구 수동 변경 |
| `/api/v1/live-match/players` | 실시간 계산 선수 확정 |
| `/api/v1/youtube/live/scoreboard/reset` | 점수·연속득점 OCR 상태 초기화 |

---

## 4. 상태 및 선수 API

### 4.1 `GET /api/v1/health`

서버가 로드한 샷 데이터와 모델을 확인합니다.

#### 응답 예시

```json
{
  "ok": true,
  "records": 842,
  "modelVersion": "symmetric-catboost-v2",
  "engineVersion": "symmetric-hybrid-v2",
  "prematchSource": "postgres"
}
```

| 필드 | 설명 |
|---|---|
| `records` | 로드한 유효 샷 레코드 수 |
| `modelVersion` | 좌표 모델 버전 |
| `engineVersion` | 하이브리드 엔진 버전 |
| `prematchSource` | `postgres`, `csv` 등 경기 전 데이터 소스 |

---

### 4.2 `GET /api/v1/prematch/players`

#### Query Parameters

| 이름 | 필수 | 기본값 | 설명 |
|---|---:|---|---|
| `league` | N | `PBA` | `PBA` 또는 `LPBA` |
| `active_only` | N | `true` | `false`이면 비활성 선수도 포함 |

#### 요청

```http
GET /api/v1/prematch/players?league=PBA&active_only=true
```

#### 응답 예시

```json
{
  "league": "PBA",
  "seasonCode": 2026,
  "dataSource": "postgres",
  "players": [
    {
      "code": "M0017784",
      "name": "선수명",
      "shortName": "선수명",
      "league": "PBA",
      "activeRoster": true,
      "imageIsPlaceholder": false,
      "imageUrl": "/api/v1/players/M0017784/image?league=PBA"
    }
  ]
}
```

---

### 4.3 `GET /api/v1/players/{player_code}/image`

#### Query Parameters

| 이름 | 기본값 | 설명 |
|---|---|---|
| `league` | `PBA` | 선수 리그 |

이미지가 있으면 저장된 MIME 타입으로 바이너리를 반환합니다. 없으면 다음을 반환합니다.

```json
{
  "error": "image_not_found"
}
```

---

## 5. 경기 전 승률 API

### 5.1 `POST /api/v1/match-probability`

#### 요청

```json
{
  "league": "PBA",
  "season_code": 2026,
  "player_a_code": "M0017784",
  "player_b_code": "M0017160"
}
```

| 필드 | 필수 | 설명 |
|---|---:|---|
| `league` | Y | `PBA` 또는 `LPBA` |
| `season_code` | Y | 현재 서비스 기준 `2026` |
| `player_a_code` | Y | 선수 A 코드 |
| `player_b_code` | Y | 선수 B 코드 |

#### 응답 주요 구조

```json
{
  "modelVersion": "cuecast-prematch-linear-v1",
  "predictionMethod": "confidence-adjusted-linear",
  "league": "PBA",
  "seasonCode": 2026,
  "displayLabel": "A_ADVANTAGE",
  "playerA": {
    "code": "M0017784",
    "name": "선수 A",
    "shortName": "선수 A",
    "winProbability": 0.613,
    "elo": 1570.4,
    "career": {
      "matches": 40,
      "wins": 25,
      "losses": 15
    },
    "season": {
      "matches": 0,
      "wins": 0,
      "losses": 0
    },
    "recent": {
      "last5Matches": 5,
      "last5Wins": 3,
      "last10Matches": 10,
      "last10Wins": 6
    },
    "performanceScore": 0.35,
    "performanceInnings": 200,
    "metrics": {
      "AVG": 1.5,
      "TS": 55.0,
      "BRS": 48.2,
      "5HS": 13.1,
      "HR": 10
    },
    "imageUrl": "/api/v1/players/M0017784/image?league=PBA",
    "imageIsPlaceholder": false
  },
  "playerB": {
    "winProbability": 0.387
  },
  "componentProbabilities": {
    "elo": 0.61,
    "career": 0.58,
    "season": 0.5,
    "recent": 0.56,
    "performance": 0.64
  },
  "componentConfidences": {
    "elo": 1.0,
    "career": 1.0,
    "season": 0.0,
    "recent": 1.0,
    "performance": 1.0
  },
  "baseWeights": {
    "elo": 0.3,
    "career": 0.175,
    "season": 0.05,
    "recent": 0.025,
    "performance": 0.45
  },
  "finalWeights": {},
  "confidence": {
    "score": 0.95,
    "level": "high"
  },
  "keyFactors": [],
  "headToHead": {},
  "headToHeadIncludedInProbability": false
}
```

#### 규칙

- 같은 선수 코드는 거부합니다.
- PBA와 LPBA 선수는 교차 계산하지 않습니다.
- A/B 승률 합은 1입니다.
- 상대 전적은 반환하지만 현재 최종 확률에는 포함하지 않습니다.

---

## 6. 샷 성공률 및 검출 API

### 6.1 `POST /api/v1/shot-probability`

#### 요청

```json
{
  "shooter": "white",
  "before": {
    "white": [0.5583, 0.9381],
    "yellow": [0.2336, 0.7072],
    "red": [0.9643, 0.3434]
  },
  "position_error_mm": 25,
  "prediction_id": "optional-client-id"
}
```

| 필드 | 필수 | 설명 |
|---|---:|---|
| `shooter` | Y | `white` 또는 `yellow` |
| `before.white` | Y | 흰 공 `[x, y]` |
| `before.yellow` | Y | 노란 공 `[x, y]` |
| `before.red` | Y | 빨간 공 `[x, y]` |
| `position_error_mm` | N | 좌표 오차, 기본 25mm |
| `prediction_id` | N | 호출 추적용 ID |

#### 응답 대표 필드

```json
{
  "successProbability": 0.426,
  "difficulty": "보통",
  "confidence": "medium",
  "roles": {
    "cue": "white",
    "object1": "red",
    "object2": "yellow"
  },
  "dataRecords": 842
}
```

실제 응답에는 모델 구성 확률, 유사 배치, Grid, 좌표 민감도와 설명 필드가 추가될 수 있습니다.

---

### 6.2 `POST /api/v1/detection`

외부 검출기가 전달한 배치를 계산하고 최신 detection store에 저장합니다.

#### 요청

`/api/v1/shot-probability`와 같은 좌표 형식을 사용합니다.

#### 응답

HTTP `201`과 함께 최신 `version`, `before`, `shooter`, `prediction`을 반환합니다.

---

### 6.3 `GET /api/v1/detection/latest`

최신 공 배치, 확정 결과, 점수판과 샷 성공률을 조회합니다.

#### 응답 주요 필드

```json
{
  "version": 14,
  "before": {
    "white": [0.55, 0.91],
    "yellow": [0.23, 0.70],
    "red": [0.96, 0.34]
  },
  "shooter": "white",
  "prediction": {},
  "confirmedVersion": 8,
  "confirmedBefore": {},
  "confirmedPrediction": {},
  "confirmedShooter": "white",
  "analysis": {
    "confirmed": true,
    "layoutSource": "stopped"
  },
  "scoreboard": {
    "player1Score": 4,
    "player2Score": 3,
    "activeColor": "white"
  }
}
```

`confirmedVersion`은 서버 재시작·일시정지 이후에도 브라우저가 새 확정 배치를 구분하는 데 사용합니다.

---

## 7. YouTube 분석 API

> 현재 YouTube 분석 endpoint와 PostgreSQL 선수 데이터 조회를 함께 사용하는 통합 서비스는 로컬 실행에서 검증했습니다. 배포 환경에서는 YouTube 또는 DB 접근이 거부될 수 있으며, 이 경우 HTTP 오류와 워커의 `error` 상태를 확인해야 합니다. 점수판 OCR에는 시스템에 설치된 Tesseract 실행 프로그램이 필요합니다.

### 7.1 `POST /api/v1/youtube/info`

#### 요청

```json
{
  "url": "https://www.youtube.com/watch?v=XXXX"
}
```

#### 응답

```json
{
  "title": "PBA 경기 영상",
  "durationSeconds": 5324.2,
  "sourceKind": "youtube"
}
```

---

### 7.2 `POST /api/v1/youtube/analyze`

특정 시점 이전 구간에서 마지막 안정 배치를 찾아 한 번 분석합니다.

#### 요청

```json
{
  "url": "https://www.youtube.com/watch?v=XXXX",
  "timestamp_seconds": 840.0,
  "lookback_seconds": 12.0,
  "shooter": "white",
  "position_error_mm": 25
}
```

#### 응답

HTTP `201`과 함께 배치, 영상 분석 메타데이터와 샷 확률을 반환하고 최신 detection store에 저장합니다.

---

### 7.3 `POST /api/v1/youtube/live/start`

#### 요청

```json
{
  "url": "https://www.youtube.com/watch?v=XXXX",
  "timestamp_seconds": 0,
  "shooter": "white"
}
```

- 기존 detection과 실시간 승률 상태를 초기화합니다.
- `shooter`는 `white` 또는 `yellow`만 허용합니다.
- HTTP `202`와 워커 상태를 반환합니다.

---

### 7.4 `GET /api/v1/youtube/live/status`

#### 응답 대표 필드

```json
{
  "state": "running",
  "url": "https://www.youtube.com/watch?v=XXXX",
  "timestampSeconds": 125.4,
  "detail": "분석 중"
}
```

워커 구현에 따라 진행 프레임, 오류, 점수판 상태와 내부 메타데이터가 추가될 수 있습니다.

---

### 7.5 `POST /api/v1/youtube/live/stop`

#### 요청

```json
{}
```

현재 분석 워커를 중지하고 최신 상태를 반환합니다.

---

### 7.6 `POST /api/v1/youtube/live/sync`

#### 요청

```json
{
  "timestamp_seconds": 900.0
}
```

브라우저가 이동한 재생 위치에 분석 워커를 맞춥니다. HTTP `202`를 반환합니다.

---

### 7.7 `POST /api/v1/youtube/live/shooter`

#### 요청

```json
{
  "shooter": "yellow"
}
```

테스트 또는 OCR 확정 전 수구를 수동 변경합니다.

---

### 7.8 `POST /api/v1/youtube/live/scoreboard/reset`

#### 요청

```json
{}
```

선수 선택은 유지하고 다음 필드를 `null`로 초기화합니다.

- `player1Score`
- `player2Score`
- `player1Run`
- `player2Run`

---

## 8. 실시간 세트 승률 API

### 8.1 `POST /api/v1/live-match/players`

실시간 승률 계산에 사용할 두 선수의 정식 이름을 지정합니다.

#### 요청

```json
{
  "player_a": "선수 A 정식 이름",
  "player_b": "선수 B 정식 이름"
}
```

선수를 해제할 때는 `null`을 전달할 수 있습니다.

#### 응답 상태

```json
{
  "state": "waiting",
  "detail": "점수판 인식 대기 중",
  "prematch": {
    "playerA": {},
    "playerB": {},
    "prematchMatchProbabilityA": 0.61
  },
  "result": null
}
```

선수 이름만 확정되어도 `prematch` 미리보기를 반환합니다.

---

### 8.2 `GET /api/v1/live-match-probability/latest`

#### 대기 응답

```json
{
  "state": "waiting",
  "detail": "현재 포메이션 성공률 대기 중",
  "prematch": {},
  "result": null
}
```

#### 계산 완료 응답 대표 구조

```json
{
  "state": "ready",
  "detail": "계산 완료",
  "result": {
    "probabilityScope": "current_set",
    "playerA": {},
    "playerB": {},
    "setWinProbabilityA": 0.573,
    "setWinProbabilityB": 0.427,
    "prematchSource": "postgres",
    "dataSource": "server_db",
    "playerNameSource": "manual",
    "inputs": {
      "scoreA": 4,
      "scoreB": 3,
      "setsToWin": 1
    }
  }
}
```

#### 계산 조건

다음 항목이 모두 준비되어야 `ready`가 됩니다.

1. 선수 A·B 정식 이름
2. 두 선수 현재 점수
3. 현재 수구와 선수 매핑
4. 확정된 현재 샷 성공률

`probabilityScope=current_set`이므로 전체 매치 승률로 해석하지 않습니다.

---

## 9. UI 및 정적 경로
