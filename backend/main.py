# FastAPI 앱 + 라우팅 (= Spring 의 Controller)
# 실행:  cd backend && ../venv/bin/uvicorn main:app --reload --port 8000
# 문서:  http://localhost:8000/docs  (Swagger UI 자동 생성)
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import services
from schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(title="Billiard Analysis API", version="0.1.0")

# React(다른 포트)에서 호출하려면 CORS 필요 — Spring 의 @CrossOrigin 대응
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/videos")
def get_videos():
    """분석 완료된 영상 목록 (+턴 수)."""
    return services.list_videos()


@app.get("/videos/{video_id}/turns")
def get_turns(video_id: str):
    """영상의 전체 턴 데이터 (before/after 좌표, 수구, 성공 여부, 시간 구간)."""
    turns = services.load_turns(video_id)
    if not turns:
        raise HTTPException(404, f"분석 결과 없음: {video_id} (먼저 /analyze 로 분석)")
    return turns

@app.get("/videos/{video_id}/status")
def get_status(video_id: str):
    """분석 상태 폴링용 — queued → running → done | failed"""
    return services.get_status(video_id)

@app.get("/videos/{video_id}/turn-at")
def get_turn_at(video_id: str, time: float):
    """재생시간(초)의 턴 + 그 배치의 성공 확률 (프론트 동기화용 메인 엔드포인트)."""
    rec = services.turn_at(video_id, time)
    if rec is None:
        return {"turn": None, "message": "이 시점 이전에 턴 없음"}
    prob = services.shot_probability(rec["shooter"], rec["before"])
    return {**rec, "probability": prob}   # prob 이 None 이면 엔진 꺼진 상태


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, bg: BackgroundTasks):
    """유튜브 URL 분석 시작 (다운로드→추출→S3). 완료 여부는 /videos/{id}/turns 로 폴링."""
    if not req.url.strip():
        raise HTTPException(400, "url 이 비어 있음")
    vid = services.enqueue_analysis(req.url.strip())
    return AnalyzeResponse(status="queued", video_id=vid,
                           detail="분석 시작됨 — 완료까지 영상 길이에 따라 수 분 소요")
