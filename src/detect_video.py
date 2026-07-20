# 영상 파이프라인: 동영상 → 프레임별 YOLO 공 탐지 → 당구대 좌표 변환
# 출력: 탐지 오버레이+미니맵 합성 영상(out_video.mp4), 프레임별 좌표(coords.csv)
# 사용법: venv/bin/python detect_video.py 영상경로 [--every N] [--max-frames N]
import argparse
import csv
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

from detect_pipeline import find_table_corners, px_to_table
from simulate import TABLE_H, TABLE_W, estimate_probability, pick_robust_shot

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_3cls.pt")
BGR = {"white": (255, 255, 255), "yellow": (0, 180, 255), "red": (0, 0, 220)}


# ---------- 탑뷰(정면 카메라) 판별 ----------
# 방송 중계는 측면 카메라로 전환되는 구간이 있어서, 기준 꼭짓점과 어긋난
# 프레임은 좌표 변환에서 제외해야 한다.
def detect_corners_fast(frame, scale=0.25):
    """축소본에서 당구대 꼭짓점 검출 (프레임별 검증용). 실패 시 None."""
    small = cv2.resize(frame, None, fx=scale, fy=scale)
    try:
        # 커널도 축소 비율에 맞춰야 주변 그래픽(점수판 등)과 붙지 않는다
        return find_table_corners(small, close_ks=max(3, round(15 * scale))) / scale
    except (ValueError, cv2.error):
        return None


def plausible_top_view(corners, frame_shape):
    """꼭짓점이 탑뷰다운 모양(가로 2:1, 수평 변, 적정 크기, 경계 미접촉)인지 확인.

    경계 접촉 검사가 중요하다: 탑뷰 당구대는 항상 화면 안쪽에 여백을 두고 잡히므로,
    화면 가장자리에 걸친 사각형(회색 안내판, 초록 바닥 등 배경 오브젝트)은 비율이
    우연히 2:1이어도 당구대가 아니다."""
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
    m_x, m_y = 0.01 * w_img, 0.01 * h_img
    inside = (corners[:, 0].min() > m_x and corners[:, 0].max() < w_img - m_x
              and corners[:, 1].min() > m_y and corners[:, 1].max() < h_img - m_y)
    return 1.6 < aspect < 2.4 and horizontal and 0.2 < area < 0.8 and inside


# ---------- 큐대 방향 검출 ----------
class CueDetector:
    """정지 배치 중 프레임에서 큐대(긴 직선)를 찾아 조준 방향을 추정한다.
    직선의 연장선이 수구(white/yellow) 가까이 지나면 그 공을 치는 것으로 판정.
    최근 프레임 다수결+원형 평균으로 안정화. 반환: (수구 이름, 조준 각도 deg)"""
    LINE_DIST = 28      # 직선-수구 중심 허용 거리 (px)
    TIP_MIN, TIP_MAX = 20, 450   # 큐 끝-수구 거리 범위 (px)

    def __init__(self):
        self.hist = deque(maxlen=7)

    def reset(self):
        self.hist.clear()

    def _detect(self, frame, corners, balls_px):
        scale = 0.5
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 60, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                                minLineLength=120, maxLineGap=8)
        if lines is None:
            return None

        cand = None  # (직선-공 거리, 수구 이름, 큐 뒤끝 px, 공 px)
        for x1, y1, x2, y2 in lines.reshape(-1, 4) / scale:
            p1, p2 = np.array([x1, y1]), np.array([x2, y2])
            seg = p2 - p1
            seg_len = np.linalg.norm(seg)
            if seg_len < 1:
                continue
            for name, ball in balls_px.items():
                ball = np.array(ball)
                # 공 중심에서 직선까지 수직 거리 (2D 외적)
                rel = ball - p1
                d_line = abs(seg[0] * rel[1] - seg[1] * rel[0]) / seg_len
                if d_line > self.LINE_DIST:
                    continue
                d1, d2 = np.linalg.norm(ball - p1), np.linalg.norm(ball - p2)
                tip, butt = (p1, p2) if d1 < d2 else (p2, p1)
                d_tip = min(d1, d2)
                if not (self.TIP_MIN < d_tip < self.TIP_MAX):
                    continue
                # 큐는 공 바깥에서 공을 향해야 함 (공이 선분 내부에 있으면 제외)
                if np.dot(ball - tip, tip - butt) < 0:
                    continue
                if cand is None or d_line < cand[0]:
                    cand = (d_line, name, butt, ball)
        if cand is None:
            return None

        _, name, butt, ball = cand
        # 픽셀 방향 → 테이블(미터) 방향으로 변환해 각도 계산
        (bx, by), (tx, ty) = px_to_table([butt, ball], corners)
        dx, dy = (tx - bx) * TABLE_W, (ty - by) * TABLE_H
        return name, float(np.degrees(np.arctan2(dy, dx)) % 360)

    def update(self, frame, corners, balls_px):
        self.hist.append(self._detect(frame, corners, balls_px))
        dets = [d for d in self.hist if d]
        if len(dets) < 3:
            return None
        ball = Counter(d[0] for d in dets).most_common(1)[0][0]
        angs = np.deg2rad([d[1] for d in dets if d[0] == ball])
        mean = np.degrees(np.arctan2(np.sin(angs).mean(), np.cos(angs).mean())) % 360
        return ball, float(mean)


# ---------- 샷 직전 배치 감지 + 확률 계산 (시뮬레이션 엔진 통합) ----------
class LayoutTracker:
    """공 3개가 모두 멈춘 '샷 직전 배치'를 감지하고, 배치가 바뀔 때마다
    시뮬레이션으로 3쿠션 성공 확률(white/yellow 수구 각각)을 계산한다."""
    STILL_WIN = 25      # 정지 판정에 쓰는 최근 관측 수
    STILL_EPS = 0.006   # 윈도우 안 이동 허용치 (정규화 좌표, 약 1.7cm)
    SEEN_WITHIN = 40    # 이 프레임 수 안에 관측된 공만 유효
    CHANGE_EPS = 0.02   # 이보다 크게 움직였으면 새 배치로 판정

    def __init__(self, n_angles=120):   # 스핀 그리드(3x3) 추가로 샷 수가 늘어 각도는 3° 간격
        self.n_angles = n_angles
        self.hist = {b: deque(maxlen=self.STILL_WIN) for b in ("white", "yellow", "red")}
        self.last_layout = None
        self.probs = None
        self.best = {}      # 수구별 추천 샷 (성공 각도 폭 최대): (각도, 속도, side, vert, ±폭)
        self.fans = {}      # 수구별 성공 각도 전체 목록 (부채살 표시용)
        self.shots = {}     # 수구별 성공 샷 전체 (조준 방향 조건부 확률 집계용)
        self.records = []   # (frame, layout, p_white, p_yellow)

    def update(self, frame_idx, balls_tbl):
        """정지 배치면 (p_white, p_yellow) 반환, 진행 중이면 None."""
        best = {}
        for name, conf, tx, ty in balls_tbl:   # 클래스별 최고 신뢰도 1개만
            if name in self.hist and conf > best.get(name, (0,))[0]:
                best[name] = (conf, tx, ty)
        for name, (_, tx, ty) in best.items():
            self.hist[name].append((frame_idx, tx, ty))

        layout = {}
        for b, dq in self.hist.items():
            if len(dq) < 8 or frame_idx - dq[-1][0] > self.SEEN_WITHIN:
                return None                     # 관측 부족 또는 오래 안 보임
            xs = [p[1] for p in dq]
            ys = [p[2] for p in dq]
            if max(xs) - min(xs) > self.STILL_EPS or max(ys) - min(ys) > self.STILL_EPS:
                return None                     # 아직 움직이는 중
            layout[b] = (xs[-1], ys[-1])

        changed = self.last_layout is None or any(
            abs(layout[b][0] - self.last_layout[b][0])
            + abs(layout[b][1] - self.last_layout[b][1]) > self.CHANGE_EPS
            for b in layout)
        if changed:
            p_w, shots_w = estimate_probability(layout, cue="white", n_angles=self.n_angles)[:2]
            p_y, shots_y = estimate_probability(layout, cue="yellow", n_angles=self.n_angles)[:2]
            self.last_layout = layout
            self.probs = (p_w, p_y)
            step = 360.0 / self.n_angles
            for cue_name, shots in (("white", shots_w), ("yellow", shots_y)):
                self.best[cue_name], self.fans[cue_name] = pick_robust_shot(shots, step)
                self.shots[cue_name] = shots
            self.records.append((frame_idx, layout, p_w, p_y))
            print(f"  프레임 {frame_idx}: 새 배치 감지 → white {p_w:.1%} / yellow {p_y:.1%}")
        return self.probs

    def aim_prob(self, cue_name, aim_deg, tol=9.0):
        """저장된 성공 샷에서 조준 방향 ±tol 윈도우의 성공 비율 집계 (재시뮬레이션 없음)."""
        step = 360.0 / self.n_angles
        n_win = (2 * int(tol / step) + 1) * 27      # 윈도우 내 각도 샘플 x (힘3 x 사이드3 x 상하3)
        hits = sum(1 for s in self.shots.get(cue_name, [])
                   if min((s[0] - aim_deg) % 360, (aim_deg - s[0]) % 360) <= tol)
        return hits / n_win


def render_prediction_pane(balls_tbl, tracker, top_view, still, cue_aim, aim_p, w, h):
    """우측 예측 화면: 0~1 정규화 테이블 위에 공 위치·좌표·확률·추천 샷 표시.
    still=False(공 이동 중)면 확률·부채살 대신 in play 표시.
    cue_aim=(수구, 각도)가 있으면 조준 방향과 그 방향의 조건부 확률 표시."""
    pane = np.full((h, w, 3), (35, 35, 35), np.uint8)
    tw = min(w - 80, (h - 200) * 2)          # 테이블은 2:1 비율 유지
    th = tw // 2
    x0, y0 = (w - tw) // 2, 40
    cv2.rectangle(pane, (x0, y0), (x0 + tw, y0 + th), (140, 90, 30), -1)
    cv2.rectangle(pane, (x0, y0), (x0 + tw, y0 + th), (255, 255, 255), 3)

    to_px = lambda tx, ty: (x0 + int(tx * tw), y0 + int(ty * th))
    r = max(6, int(tw * 0.0108))             # 공 반지름 (실제 비율)

    # 성공 방향 부채살: 3쿠션이 되는 모든 각도를 반투명 광선으로 표시
    if top_view and still and tracker.probs and tracker.last_layout:
        overlay = pane.copy()
        for cue_name in ("white", "yellow"):
            if cue_name not in tracker.last_layout:
                continue
            p1 = to_px(*tracker.last_layout[cue_name])
            for a in tracker.fans.get(cue_name) or []:
                rad = np.deg2rad(a)
                p2 = (int(p1[0] + 0.10 * tw * np.cos(rad)),
                      int(p1[1] + 0.10 * tw * np.sin(rad)))
                cv2.line(overlay, p1, p2, BGR[cue_name], 2)
        pane = cv2.addWeighted(overlay, 0.35, pane, 0.65, 0)

    for name, _, tx, ty in balls_tbl:
        p = to_px(tx, ty)
        cv2.circle(pane, p, r, BGR.get(name, (0, 255, 0)), -1)
        cv2.circle(pane, p, r, (30, 30, 30), 2)
        cv2.putText(pane, f"({tx:.2f}, {ty:.2f})", (p[0] + r + 6, p[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    ty_text = y0 + th + 45
    if not top_view:
        cv2.putText(pane, "CAMERA CUT - coords excluded", (x0, ty_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
    elif still and tracker.probs:
        cv2.putText(pane, f"3C prob  W {tracker.probs[0]:.1%} | Y {tracker.probs[1]:.1%}",
                    (x0, ty_text), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 230, 80), 2)
        for row, cue_name in enumerate(("white", "yellow")):
            shot = tracker.best.get(cue_name)
            if not shot or cue_name not in (tracker.last_layout or {}):
                continue
            deg, v, s, t, tol = shot          # 성공 각도 폭이 가장 넓은 샷 (±tol)
            cv2.putText(pane, f"{cue_name[0].upper()} best {deg:.0f}deg(+-{tol:.0f}) "
                              f"v{v:.1f} side{s:+.1f} vert{t:+.1f}",
                        (x0, ty_text + 35 + row * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR[cue_name], 2)
            bx, by = tracker.last_layout[cue_name]
            p1 = to_px(bx, by)
            rad = np.deg2rad(deg)             # 2:1 비율이 유지되므로 각도 왜곡 없음
            p2 = (int(p1[0] + 0.13 * tw * np.cos(rad)),
                  int(p1[1] + 0.13 * tw * np.sin(rad)))
            cv2.arrowedLine(pane, p1, p2, BGR[cue_name], 3, tipLength=0.25)

        # 검출된 큐대 조준 방향 + 그 방향의 조건부 확률 (마젠타)
        if cue_aim and aim_p is not None and cue_aim[0] in (tracker.last_layout or {}):
            name, deg = cue_aim
            p1 = to_px(*tracker.last_layout[name])
            rad = np.deg2rad(deg)
            p2 = (int(p1[0] + 0.16 * tw * np.cos(rad)),
                  int(p1[1] + 0.16 * tw * np.sin(rad)))
            cv2.arrowedLine(pane, p1, p2, (255, 0, 255), 3, tipLength=0.2)
            cv2.putText(pane, f"AIM {name[0].upper()} {deg:.0f}deg -> {aim_p:.1%}",
                        (x0, ty_text + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        cv2.putText(pane, "in play...", (x0, ty_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    return pane


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
    m_inv = None            # 테이블 정규화 좌표 → 픽셀 역변환 (조준선 표시용)
    tracker = LayoutTracker()
    cue_detector = CueDetector()
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
                m_inv = cv2.getPerspectiveTransform(
                    np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), corners)
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
            cx, cy, bw, bh = box.xywh[0].tolist()
            balls.append((name, conf, cx, cy, bw, bh))

        if balls and top_view:
            tbl = px_to_table([[b[2], b[3]] for b in balls], corners)
            balls_tbl = [(n, c, float(tx), float(ty))
                         for (n, c, *_), (tx, ty) in zip(balls, tbl)]
        else:
            balls_tbl = []

        for name, conf, tx, ty in balls_tbl:
            rows.append([frame_idx, round(frame_idx / fps, 3), name,
                         round(conf, 3), round(tx, 4), round(ty, 4)])

        # 정지 배치 감지 → 시뮬레이션 확률 (탑뷰에서만). 반환값이 있으면 정지 상태
        still = bool(tracker.update(frame_idx, balls_tbl)) if top_view else False

        # 정지 중에만 큐대 방향 검출 → 조준 방향의 조건부 확률
        cue_aim, aim_p = None, None
        if still:
            balls_px = {}
            for name, conf, cx, cy, *_ in balls:
                if name in ("white", "yellow") and conf > balls_px.get(name, (0,))[0]:
                    balls_px[name] = (conf, cx, cy)
            cue_aim = cue_detector.update(frame, corners,
                                          {n: v[1:] for n, v in balls_px.items()})
            if cue_aim:
                aim_p = tracker.aim_prob(*cue_aim)
        else:
            cue_detector.reset()

        # 좌측: 원본 영상 + 공 박스 (신뢰도 대신 0~1 정규화 좌표 표시)
        annotated = frame.copy()
        for k, (name, _, cx, cy, bw, bh) in enumerate(balls):
            color = BGR.get(name, (0, 255, 0))
            p1 = (int(cx - bw / 2) - 4, int(cy - bh / 2) - 4)
            p2 = (int(cx + bw / 2) + 4, int(cy + bh / 2) + 4)
            cv2.rectangle(annotated, p1, p2, color, 3)
            label = (f"({balls_tbl[k][2]:.2f}, {balls_tbl[k][3]:.2f})"
                     if k < len(balls_tbl) else name)
            org = (p1[0], p1[1] - 14)
            cv2.putText(annotated, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(annotated, label, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if not top_view:
            cv2.putText(annotated, "camera cut - coords excluded", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        # 검출된 조준선을 원본에도 그려 실제 큐대와 겹치는지 확인 가능하게
        if cue_aim and m_inv is not None and cue_aim[0] in (tracker.last_layout or {}):
            name, deg = cue_aim
            bx, by = tracker.last_layout[name]
            rad = np.deg2rad(deg)
            end = ((bx * TABLE_W + 0.6 * np.cos(rad)) / TABLE_W,
                   (by * TABLE_H + 0.6 * np.sin(rad)) / TABLE_H)
            pts = cv2.perspectiveTransform(
                np.float32([[bx, by], end]).reshape(-1, 1, 2), m_inv).reshape(-1, 2)
            cv2.arrowedLine(annotated, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)),
                            (255, 0, 255), 4, tipLength=0.2)

        # 좌우 1:1 합성 (좌: 당구 영상, 우: 예측 화면)
        out_h = 720
        left_w = int(round(annotated.shape[1] * out_h / annotated.shape[0] / 2)) * 2
        left = cv2.resize(annotated, (left_w, out_h))
        right = render_prediction_pane(balls_tbl, tracker, top_view, still,
                                       cue_aim, aim_p, left_w, out_h)
        composed = np.hstack([left, right])

        if writer is None:
            h, w = composed.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / args.every, (w, h))
        writer.write(composed)
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

    with open("shots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "p_white", "p_yellow",
                    "white_x", "white_y", "yellow_x", "yellow_y", "red_x", "red_y"])
        for fi, layout, p_w, p_y in tracker.records:
            w.writerow([fi, round(fi / fps, 2), round(p_w, 4), round(p_y, 4)]
                       + [round(v, 4) for b in ("white", "yellow", "red") for v in layout[b]])

    print(f"\n처리: {n_proc}프레임 in {elapsed:.1f}s ({n_proc / max(elapsed, 1e-9):.1f} fps)")
    print(f"탑뷰 프레임: {n_top}/{n_proc} ({n_top / max(n_proc, 1):.1%}) — 나머지는 카메라 전환 구간으로 제외")
    print(f"감지된 정지 배치: {len(tracker.records)}개")
    print(f"저장: {args.out} (합성 영상), {args.csv} (좌표 {len(rows)}행), shots.csv (배치별 확률)")


if __name__ == "__main__":
    main()
