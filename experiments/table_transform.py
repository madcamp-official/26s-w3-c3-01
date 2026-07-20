# 13~15단계: 당구대 자동 검출 → 원근 변환 → 공 좌표 정규화 (로컬 검증용)
import cv2
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-parkminsu-madcamp-03w/78983694-7ef5-43e9-99f9-722555bb83c8/scratchpad"
img = cv2.imread(f"{SCRATCH}/pba_frame.png")
H_img, W_img = img.shape[:2]
print(f"이미지 크기: {W_img}x{H_img}")

# ---------- 13단계: 파란 영역으로 당구대 꼭짓점 자동 검출 ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([90, 80, 80])
upper = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
table_cnt = max(cnts, key=cv2.contourArea)
print(f"당구대 면적: {cv2.contourArea(table_cnt):.0f}px² ({cv2.contourArea(table_cnt)/(W_img*H_img)*100:.1f}% of frame)")

hull = cv2.convexHull(table_cnt)
eps = 0.02 * cv2.arcLength(hull, True)
quad = cv2.approxPolyDP(hull, eps, True)
print(f"근사 꼭짓점 개수: {len(quad)}")
if len(quad) != 4:
    quad = cv2.boxPoints(cv2.minAreaRect(table_cnt))
    print("→ minAreaRect 대체 사용")

def order_corners(pts):
    pts = np.array(pts).reshape(-1, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)],
                       pts[np.argmax(s)], pts[np.argmax(d)]])

corners = order_corners(quad)
print("꼭짓점 (좌상/우상/우하/좌하):")
for c in corners:
    print(f"  ({c[0]:.0f}, {c[1]:.0f})")

# 가로/세로 비율 확인 (실제 당구대는 2:1)
w_top = np.linalg.norm(corners[1] - corners[0])
h_left = np.linalg.norm(corners[3] - corners[0])
print(f"검출된 가로/세로 비율: {w_top/h_left:.2f} (기대값 ≈ 2.0)")

# ---------- 14단계: 시각화 ----------
vis = img.copy()
for i, p in enumerate(corners):
    cv2.circle(vis, tuple(p.astype(int)), 18, (0, 0, 255), -1)
    cv2.putText(vis, str(i), tuple((p + 25).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

W, Ht = 800, 400
M_view = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [W, 0], [W, Ht], [0, Ht]]))
warp = cv2.warpPerspective(img, M_view, (W, Ht))

cv2.imwrite(f"{SCRATCH}/out_mask.png", mask)
cv2.imwrite(f"{SCRATCH}/out_corners.png", vis)
cv2.imwrite(f"{SCRATCH}/out_warped.png", warp)

# ---------- 15단계: 공 자동 검출(색 기반, YOLO 대용) + 좌표 변환 ----------
def px_to_table(points_px, corners):
    dst = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    M = cv2.getPerspectiveTransform(corners, dst)
    pts = np.float32(points_px).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)

# 당구대 내부에서 "파란색이 아닌" 작은 원형 덩어리 = 공
table_mask = np.zeros(mask.shape, np.uint8)
cv2.fillPoly(table_mask, [corners.astype(int)], 255)
table_mask = cv2.erode(table_mask, np.ones((25, 25), np.uint8))  # 쿠션 가장자리 제외

not_blue = cv2.bitwise_and(cv2.bitwise_not(mask), table_mask)
not_blue = cv2.morphologyEx(not_blue, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

ball_cnts, _ = cv2.findContours(not_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
balls = []
for c in ball_cnts:
    area = cv2.contourArea(c)
    if not (300 < area < 5000):          # 공 크기 범위만
        continue
    peri = cv2.arcLength(c, True)
    circularity = 4 * np.pi * area / (peri * peri)
    if circularity < 0.6:                 # 원형이 아니면 제외 (큐대, 팔 등)
        continue
    (cx, cy), r = cv2.minEnclosingCircle(c)
    # 색 분류: 공 중심부의 평균 HSV
    patch = hsv[int(cy - r/2):int(cy + r/2), int(cx - r/2):int(cx + r/2)]
    h_mean, s_mean, v_mean = patch[..., 0].mean(), patch[..., 1].mean(), patch[..., 2].mean()
    if s_mean < 90 and v_mean > 150:
        color = "white"
    elif h_mean < 12 or h_mean > 165:
        color = "red"
    else:
        color = "yellow"   # 주황빛 노란 큐볼
    balls.append((color, cx, cy))

print(f"\n검출된 공: {len(balls)}개")
tbl = px_to_table([[b[1], b[2]] for b in balls], corners)
for (color, cx, cy), (tx, ty) in zip(balls, tbl):
    print(f"  {color:6s}: 픽셀({cx:6.0f}, {cy:6.0f}) → 당구대({tx:.3f}, {ty:.3f})")

# ---------- 미니맵: 최종 결과물 미리보기 ----------
mini = np.full((400, 800, 3), (140, 90, 30), np.uint8)   # 파란 천 느낌 배경
cv2.rectangle(mini, (0, 0), (799, 399), (255, 255, 255), 4)
bgr = {"white": (255, 255, 255), "yellow": (0, 180, 255), "red": (0, 0, 220)}
for (color, _, _), (tx, ty) in zip(balls, tbl):
    cv2.circle(mini, (int(tx * 800), int(ty * 400)), 12, bgr[color], -1)
    cv2.circle(mini, (int(tx * 800), int(ty * 400)), 12, (30, 30, 30), 2)
cv2.imwrite(f"{SCRATCH}/out_minimap.png", mini)
print("\n결과 이미지 저장 완료: out_mask / out_corners / out_warped / out_minimap")
