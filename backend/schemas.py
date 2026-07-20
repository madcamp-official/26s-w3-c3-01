# Pydantic 모델 (= Spring 의 DTO + @Valid)
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    status: str
    video_id: str | None = None
    detail: str | None = None
