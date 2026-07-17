from roboflow import Roboflow

rf = Roboflow(api_key="Ns7fc03BlToEUh5Z6j7o")
project = rf.workspace("-ahcxq").project("-wdzpn")
version = project.version(1)
dataset = version.download("yolov11")
print(dataset.location)  # 다운로드 경로 확인