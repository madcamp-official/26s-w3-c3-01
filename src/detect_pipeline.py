# 통합 파이프라인: 영상 프레임 → YOLO 공 탐지 → 당구대 좌표 변환
# 사용법: venv/bin/python detect_pipeline.py [이미지경로]
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_frame.png"
# 모델은 이 스크립트와 같은 폴더(src/)에 있다 → 현재 작업 폴더와 무관하게 찾도록 절대경로화
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_3cls.pt")


# ---------- 당구대 꼭짓점 자동 검출 ----------
# 중계마다 천 색이 다르다: PBA 팀리그=파랑, 빌리어즈TV=어두운 초록, LPBA=회색(저채도).
# 색 범위별로 후보 사각형을 따로 뽑은 뒤 형태로 최적 후보를 고른다. 합집합 마스크를
# 쓰면 (예: 회색 테이블 주변의 초록 바닥처럼) 다른 색 영역이 최대 윤곽을 차지해
# 당구대를 놓치기 때문이다.
CLOTH_HSV_RANGES = (
    ((90, 80, 80), (130, 255, 255)),   # 파란 천
    ((65, 60, 40), (90, 255, 255)),    # 초록 천 (어두운 톤 포함)
    ((90, 8, 110), (125, 60, 190)),    # 회색 천 (푸른기 도는 저채도 중간 밝기)
)


def _quad_of(cnt):
    hull = cv2.convexHull(cnt)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(cnt))
    pts = np.array(quad).reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])  # 좌상,우상,우하,좌하


def _quad_plausible(q, shape):
    """탑뷰 당구대다운 사각형인지: 가로 2:1 비율, 수평 변, 적정 면적, 경계 미접촉."""
    h_img, w_img = shape[:2]
    top_w = np.linalg.norm(q[1] - q[0])
    bot_w = np.linalg.norm(q[2] - q[3])
    left_h = np.linalg.norm(q[3] - q[0])
    right_h = np.linalg.norm(q[2] - q[1])
    if min(top_w, bot_w, left_h, right_h) < 1:
        return False
    aspect = (top_w + bot_w) / (left_h + right_h)
    horizontal = (abs(q[1, 1] - q[0, 1]) < 0.05 * h_img
                  and abs(q[2, 1] - q[3, 1]) < 0.05 * h_img)
    area = cv2.contourArea(q) / (w_img * h_img)
    # 바닥·벽·안내판처럼 화면 가장자리에 걸친 영역 배제: 탑뷰 당구대는 항상
    # 화면 안쪽에 여백을 두고 잡힌다 (경계 접촉 0 + 면적 상한).
    m_x, m_y = 0.01 * w_img, 0.01 * h_img
    inside = (q[:, 0].min() > m_x and q[:, 0].max() < w_img - m_x
              and q[:, 1].min() > m_y and q[:, 1].max() < h_img - m_y)
    return 1.6 < aspect < 2.4 and horizontal and 0.12 < area < 0.8 and inside


def find_table_corners(img, close_ks=15):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = np.ones((close_ks, close_ks), np.uint8)
    best, best_area = None, 0.0
    fallback, fallback_area = None, 0.0
    for lo, hi in CLOTH_HSV_RANGES:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        quad = _quad_of(cnt)
        area = cv2.contourArea(quad)
        if area > fallback_area:
            fallback, fallback_area = quad, area
        if _quad_plausible(quad, img.shape) and area > best_area:
            best, best_area = quad, area
    if best is not None:
        return best
    if fallback is not None:        # 형태 검증 실패 시 기존 동작 유지 (호출부가 재검증)
        return fallback
    raise ValueError("천 색 마스크에서 윤곽을 찾지 못함")


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
