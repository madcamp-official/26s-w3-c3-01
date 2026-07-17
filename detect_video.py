# 영상 파이프라인: 동영상 → 프레임별 YOLO 공 탐지 → 당구대 좌표 변환
# 출력: 탐지 오버레이+미니맵 합성 영상(out_video.mp4), 프레임별 좌표(coords.csv)
# 사용법: venv/bin/python detect_video.py 영상경로 [--every N] [--max-frames N]
import argparse
import csv
import time

import cv2
import numpy as np
from ultralytics import YOLO

from detect_pipeline import find_table_corners, px_to_table

MODEL_PATH = "best_3cls.pt"
BGR = {"white": (255, 255, 255), "yellow": (0, 180, 255), "red": (0, 0, 220)}


# ---------- 탑뷰(정면 카메라) 판별 ----------
# 방송 중계는 측면 카메라로 전환되는 구간이 있어서, 기준 꼭짓점과 어긋난
# 프레임은 좌표 변환에서 제외해야 한다.
def detect_corners_fast(frame, scale=0.25):
    """축소본에서 당구대 꼭짓점 검출 (프레임별 검증용). 실패 시 None."""
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    try:
        return find_table_corners(small) / scale
    except (ValueError, cv2.error):
        return None


def plausible_top_view(corners, frame_shape):
    """꼭짓점이 탑뷰다운 모양(가로 2:1, 수평 변, 충분한 크기)인지 확인."""
    if corners is None:
        return False
    h_img, w_img = frame_shape[:2]
    top_w = np.linalg.norm(corners[1] - corners[0])
    bot_w = np.linalg.norm(corners[2] - corners[3])
    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])
    if min(top_w, bot_w, left_h, right_h) < 1:
        return False
    aspect = (top_w + bot_w) / (left_h + right_h)
    horizontal = (abs(corners[1, 1] - corners[0, 1]) < 0.05 * h_img
                  and abs(corners[2, 1] - corners[3, 1]) < 0.05 * h_img)
    area = cv2.contourArea(corners) / (w_img * h_img)
    return 1.6 < aspect < 2.4 and horizontal and area > 0.2


def draw_minimap(balls_tbl, w=400, h=200):
    """balls_tbl: [(이름, conf, tx, ty)] 정규화 좌표 → 미니맵 이미지"""
    mini = np.full((h, w, 3), (140, 90, 30), np.uint8)
    cv2.rectangle(mini, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    for name, conf, tx, ty in balls_tbl:
        p = (int(tx * w), int(ty * h))
        cv2.circle(mini, p, 7, BGR.get(name, (0, 255, 0)), -1)
        cv2.circle(mini, p, 7, (30, 30, 30), 1)
    return mini


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="입력 영상 경로")
    parser.add_argument("--every", type=int, default=1, help="N프레임마다 1번 처리 (기본 1=모든 프레임)")
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수 (0=전체)")
    parser.add_argument("--out", default="out_video.mp4")
    parser.add_argument("--csv", default="coords.csv")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"영상을 못 읽음: {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"입력: {args.video} ({total}프레임, {fps:.1f}fps)")

    model = YOLO(MODEL_PATH)
    corners = None          # 방송 화면에서 당구대는 고정 → 첫 프레임에서 1번만 검출
    writer = None
    rows = []
    n_proc, n_top, t0 = 0, 0, time.time()

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.every:
            continue
        if args.max_frames and n_proc >= args.max_frames:
            break

        # 프레임별 탑뷰 검증: 기준 꼭짓점과 크게 어긋나면 카메라 전환 구간
        fast = detect_corners_fast(frame)
        if corners is None:
            if plausible_top_view(fast, frame.shape):
                corners = find_table_corners(frame)   # 기준은 원본 해상도로 정밀 검출
                print(f"당구대 꼭짓점 (프레임 {frame_idx}):", corners.astype(int).tolist())
            top_view = False
        else:
            top_view = (plausible_top_view(fast, frame.shape)
                        and np.abs(fast - corners).max() < 60)

        res = model(frame, verbose=False)[0]
        balls = []
        for box in res.boxes:
            name = res.names[int(box.cls)]
            conf = float(box.conf)
            cx, cy = box.xywh[0][:2].tolist()
            balls.append((name, conf, cx, cy))

        if balls and top_view:
            tbl = px_to_table([[b[2], b[3]] for b in balls], corners)
            balls_tbl = [(n, c, float(tx), float(ty))
                         for (n, c, _, _), (tx, ty) in zip(balls, tbl)]
        else:
            balls_tbl = []

        for name, conf, tx, ty in balls_tbl:
            rows.append([frame_idx, round(frame_idx / fps, 3), name,
                         round(conf, 3), round(tx, 4), round(ty, 4)])

        # 탐지 오버레이 + 우상단 미니맵 합성
        annotated = res.plot()
        mini = draw_minimap(balls_tbl)
        mh, mw = mini.shape[:2]
        if not top_view:
            cv2.putText(mini, "CAMERA CUT", (mw // 2 - 90, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(annotated, "camera cut - coords excluded", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        annotated[10:10 + mh, annotated.shape[1] - mw - 10:annotated.shape[1] - 10] = mini

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.every, (w, h))
        writer.write(annotated)
        n_proc += 1
        n_top += top_view

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - t0
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "ball", "conf", "table_x", "table_y"])
        w.writerows(rows)

    print(f"\n처리: {n_proc}프레임 in {elapsed:.1f}s ({n_proc / max(elapsed, 1e-9):.1f} fps)")
    print(f"탑뷰 프레임: {n_top}/{n_proc} ({n_top / max(n_proc, 1):.1%}) — 나머지는 카메라 전환 구간으로 제외")
    print(f"저장: {args.out} (합성 영상), {args.csv} (좌표 {len(rows)}행)")


if __name__ == "__main__":
    main()
