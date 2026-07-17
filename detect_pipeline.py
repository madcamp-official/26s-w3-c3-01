# 통합 파이프라인: 영상 프레임 → YOLO 공 탐지 → 당구대 좌표 변환
# 사용법: venv/bin/python detect_pipeline.py [이미지경로]
import sys
import cv2
import numpy as np
from ultralytics import YOLO

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_frame.png"
MODEL_PATH = "best_3cls.pt"


# ---------- 당구대 꼭짓점 자동 검출 ----------
def find_table_corners(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([130, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(table_cnt)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(table_cnt))

    pts = np.array(quad).reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])  # 좌상,우상,우하,좌하


# ---------- 픽셀 → 당구대 정규화 좌표(0~1) ----------
def px_to_table(points_px, corners):
    dst = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    M = cv2.getPerspectiveTransform(corners, dst)
    pts = np.float32(points_px).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def main():
    img = cv2.imread(IMG_PATH)
    assert img is not None, f"이미지를 못 읽음: {IMG_PATH}"

    corners = find_table_corners(img)
    print("당구대 꼭짓점:", corners.astype(int).tolist())

    # YOLO 공 탐지
    model = YOLO(MODEL_PATH)
    res = model(IMG_PATH, verbose=False)[0]

    balls = []  # (클래스명, 신뢰도, 픽셀중심)
    for box in res.boxes:
        name = res.names[int(box.cls)]
        conf = float(box.conf)
        cx, cy = box.xywh[0][:2].tolist()
        balls.append((name, conf, cx, cy))

    print(f"\nYOLO 탐지: {len(balls)}개")
    if not balls:
        print("공을 찾지 못했습니다.")
        return

    tbl = px_to_table([[b[2], b[3]] for b in balls], corners)

    result = {}
    for (name, conf, cx, cy), (tx, ty) in zip(balls, tbl):
        print(f"  {name:6s} conf={conf:.2f}: 픽셀({cx:6.0f},{cy:6.0f}) → 당구대({tx:.3f}, {ty:.3f})")
        result[name] = (round(float(tx), 3), round(float(ty), 3))

    # 확률 모델 입력 벡터 (흰공을 큐볼로 가정한 예시)
    print("\n확률 모델 입력 벡터:", result)

    # 미니맵 시각화
    mini = np.full((400, 800, 3), (140, 90, 30), np.uint8)
    cv2.rectangle(mini, (0, 0), (799, 399), (255, 255, 255), 4)
    bgr = {"white": (255, 255, 255), "yellow": (0, 180, 255), "red": (0, 0, 220)}
    for (name, conf, _, _), (tx, ty) in zip(balls, tbl):
        p = (int(tx * 800), int(ty * 400))
        cv2.circle(mini, p, 12, bgr.get(name, (0, 255, 0)), -1)
        cv2.circle(mini, p, 12, (30, 30, 30), 2)
        cv2.putText(mini, f"{name} {conf:.2f}", (p[0] + 16, p[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite("pipeline_minimap.png", mini)

    # 원본 위에 탐지 박스 그리기
    annotated = res.plot()
    cv2.imwrite("pipeline_detect.png", annotated)
    print("저장: pipeline_detect.png (탐지 결과), pipeline_minimap.png (미니맵)")


if __name__ == "__main__":
    main()
