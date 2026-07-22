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
