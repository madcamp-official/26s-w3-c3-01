# Roboflow 데이터셋 다운로드 (일회성 유틸). 파이프라인에서 쓰지 않음.
# API 키는 코드에 박지 말고 환경변수로:
#   ROBOFLOW_API_KEY=발급받은키 venv/bin/python experiments/test1.py
import os
from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    raise SystemExit("환경변수 ROBOFLOW_API_KEY 를 설정하세요.")

rf = Roboflow(api_key=api_key)
project = rf.workspace("-ahcxq").project("-wdzpn")
version = project.version(1)
dataset = version.download("yolov11")
print(dataset.location)  # 다운로드 경로 확인
