# 턴 단위 데이터 추출: 영상 → 매 샷의 (직전 배치, 수구, 성공 여부, 이후 배치)
# 성공 판정(1순위): 방송 점수판 OCR — 샷 구간에 수구 색 점수가 +1/+2 오르면 성공.
#   점수판은 오퍼레이터가 올리는 사실상의 정답이라 궤적 분석보다 정확하다.
#   (PBA 점수판: 흰 박스=흰 수구 선수, 노란 박스=노란 수구 선수 → 수구와 바로 매칭)
# 성공 판정(폴백): 점수판을 못 찾거나 판독 불가 구간은 기존 궤적 분석 —
#   수구 궤적에서 "두 목적구 접촉 + 두 번째 접촉 전 쿠션 3회"를 직접 계산.
# 출력: turns.jsonl, turns.csv (+ --save-frames 시 qa/ 폴더에 직전/이후 프레임)
# 사용법: venv/bin/python extract_turns.py 영상경로 [--video-id ID] [--outdir DIR] [--every N]
#         [--no-scoreboard] 점수판 판정 끄기(궤적 판정만 사용)
import argparse
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from detect_pipeline import find_table_corners, px_to_table
from detect_video import detect_corners_fast, plausible_top_view
from scoreboard import ScoreReader

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_3cls.pt")
BALLS = ("white", "yellow", "red")
TABLE_W, TABLE_H = 2.84, 1.42          # 국제식 대대 경기면 (m)

CONF_MIN = 0.45         # 이 신뢰도 미만 탐지는 무시
STILL_WIN = 12          # 정지 판정에 쓰는 최근 관측 수
STILL_MIN_OBS = 8       # 정지 판정 최소 관측 수
STILL_EPS = 0.008       # 정지 판정: 윈도우 내 이동 허용치 (정규화 좌표)
SEEN_WITHIN_S = 1.5     # 이 시간 안에 관측된 공만 정지 판정에 사용
MOVE_EPS_M = 0.06       # 정지 배치에서 이만큼 벗어나면 샷 시작 (m)
MOVE_CONFIRM = 2        # 샷 시작 판정에 필요한 연속 이탈 관측 수
GAP_MAX_S = 4.0         # 탑뷰가 이 시간 이상 끊기면 클립 경계로 보고 리셋
HIT_NEAR_M = 0.22       # 접촉 판정: 목적구가 움직이기 시작할 때 수구가 이 거리 안
OBJ_MOVE_M = 0.04       # 목적구 '움직이기 시작' 판정 이동량 (m)
# 쿠션 접촉: 레일 존(레일 폭 + 공 반지름 ≈ 0.08m) 진입 횟수로 센다.
# 부호 반전 방식은 쿠션에 맞고 거의 멈추거나 레일을 따라 흐르는 반사를 놓친다.
CUSHION_ZONE_M = 0.12   # 이 거리 안이면 레일 존 진입 (= 접촉)
CUSHION_EXIT_M = 0.17   # 이 거리 밖으로 나가야 같은 레일 재접촉 인정 (히스테리시스)
CUSHION_V_MIN = 0.15    # 존 진입 시 최소 속도 (m/s) — 정지 공 지터 제거
TELEPORT_M_PER_FR = 0.3 # 프레임당 이 거리(9m/s) 이상 이동 = 편집 컷으로 판정
COVERAGE_MIN = 0.35     # 샷 구간 수구 관측 비율이 이보다 낮으면 궤적 판정 보류
CORNER_LOCK_N = 8       # 기준 꼭짓점을 이 개수의 탑뷰 프레임 합의(중앙값)로 잠금 (이상치 배제)
CORNER_CUT_TOL = 90     # 기준 대비 이 px 넘게 어긋나면 카메라 컷으로 판정
SCORE_EVERY = 5         # 점수판 샘플 주기 (처리 프레임 기준) — 탑뷰가 아니어도 읽는다
SCORE_FAIL_MARGIN_S = 3.0   # '실패' 확정에 필요한 샷 종료 후 판독 여유 (초)
REPLAY_VISIBLE_MAX = 0.3    # 샷 구간 점수판 노출 비율이 이 미만이면 리플레이 유령 턴


def to_m(x, y):
    return x * TABLE_W, y * TABLE_H


class TurnExtractor:
    """탑뷰 프레임의 공 좌표를 받아 턴(샷) 단위 레코드를 만든다."""

    def __init__(self, fps):
        self.fps = fps
        self.hist = {b: deque(maxlen=STILL_WIN) for b in BALLS}  # (frame, x, y)
        self.state = "SEEK_STILL"       # SEEK_STILL → STILL → SHOT → STILL …
        self.still_layout = None        # {ball: (x, y)} 정지 배치 (정규화)
        self.depart = {b: 0 for b in BALLS}   # 정지 배치 이탈 연속 관측 수
        self.epoch = 0                  # 클립(연속 구간) 번호 — 리셋마다 증가
        self.shot = None                # 진행 중 샷 정보
        self.turns = []
        self.last_update_frame = None

    # ---------- 내부 유틸 ----------
    def _reset(self):
        for dq in self.hist.values():
            dq.clear()
        self.state = "SEEK_STILL"
        self.still_layout = None
        self.depart = {b: 0 for b in BALLS}
        self.epoch += 1

    def _still_layout(self, frame_idx):
        """세 공 모두 정지 상태면 {ball: (x, y)} 반환, 아니면 None."""
        layout = {}
        for b, dq in self.hist.items():
            if len(dq) < STILL_MIN_OBS:
                return None
            if frame_idx - dq[-1][0] > SEEN_WITHIN_S * self.fps:
                return None
            xs = [p[1] for p in dq]
            ys = [p[2] for p in dq]
            if max(xs) - min(xs) > STILL_EPS or max(ys) - min(ys) > STILL_EPS:
                return None
            layout[b] = (float(np.median(xs)), float(np.median(ys)))
        return layout

    # ---------- 프레임 공급 ----------
    def tick(self, frame_idx, balls_tbl):
        """balls_tbl: [(이름, conf, tx, ty)] 탑뷰 프레임의 정규화 좌표.
        탑뷰가 아닌 프레임은 balls_tbl=[] 로 호출된다."""
        # 클립 경계(탑뷰 장기 끊김) 처리
        if balls_tbl:
            if (self.last_update_frame is not None
                    and frame_idx - self.last_update_frame > GAP_MAX_S * self.fps):
                if self.state == "SHOT":
                    self._finalize_shot(frame_idx, settled=False)
                self._reset()
            self.last_update_frame = frame_idx
        else:
            return

        best = {}
        for name, conf, tx, ty in balls_tbl:
            if name in self.hist and conf >= CONF_MIN and conf > best.get(name, (0,))[0]:
                best[name] = (conf, tx, ty)

        # 편집 컷 감지: 공이 물리적으로 불가능한 속도로 '순간이동'하면
        # 하이라이트 점프 컷 → 진행 중 샷을 끊고 리셋 (두 샷 병합 방지)
        for name, (_, tx, ty) in best.items():
            dq = self.hist[name]
            if dq:
                f0, x0, y0 = dq[-1]
                df = frame_idx - f0
                if 0 < df <= 6:
                    d_m = np.hypot((tx - x0) * TABLE_W, (ty - y0) * TABLE_H)
                    if d_m / df > TELEPORT_M_PER_FR:
                        if self.state == "SHOT":
                            self._finalize_shot(f0, settled=False)
                        self._reset()
                        for n2, (_, tx2, ty2) in best.items():
                            self.hist[n2].append((frame_idx, tx2, ty2))
                        return

        for name, (_, tx, ty) in best.items():
            self.hist[name].append((frame_idx, tx, ty))
            if self.state == "SHOT":
                self.shot["traj"][name].append((frame_idx, tx, ty))

        if self.state in ("SEEK_STILL", "STILL"):
            layout = self._still_layout(frame_idx)
            if self.state == "SEEK_STILL":
                if layout:
                    self.state = "STILL"
                    self.still_layout = layout
                    self.depart = {b: 0 for b in BALLS}
                return
            # STILL: 배치 갱신 + 샷 시작 감지
            if layout:
                self.still_layout = layout
            mover = self._detect_mover(best)
            if mover:
                self._start_shot(frame_idx, mover)
        elif self.state == "SHOT":
            layout = self._still_layout(frame_idx)
            if layout:
                # 세 공이 다시 정지 → 샷 종료
                self._finalize_shot(frame_idx, settled=True, layout=layout)
                self.state = "STILL"
                self.still_layout = layout
                self.depart = {b: 0 for b in BALLS}

    def flush(self, frame_idx):
        """영상 끝: 진행 중이던 샷을 마지막 관측 기준으로 마무리."""
        if self.state == "SHOT":
            self._finalize_shot(frame_idx, settled=False)

    # ---------- 샷 시작/종료 ----------
    def _detect_mover(self, best):
        """정지 배치에서 벗어난 공 감지. MOVE_CONFIRM회 연속이면 수구로 확정."""
        for b in BALLS:
            if b not in best:
                continue
            _, tx, ty = best[b]
            sx, sy = self.still_layout[b]
            dx, dy = (tx - sx) * TABLE_W, (ty - sy) * TABLE_H
            if (dx * dx + dy * dy) ** 0.5 > MOVE_EPS_M:
                self.depart[b] += 1
            else:
                self.depart[b] = 0
        confirmed = [b for b in BALLS if self.depart[b] >= MOVE_CONFIRM]
        if not confirmed:
            return None
        # 가장 먼저 이탈을 시작한(연속 이탈 수가 많은) 공이 수구
        return max(confirmed, key=lambda b: self.depart[b])

    def _start_shot(self, frame_idx, cue):
        self.shot = {
            "cue": cue,
            "before": dict(self.still_layout),
            "frame_start": frame_idx,
            "traj": {b: [] for b in BALLS},
        }
        # 수구의 이탈 관측(이미 움직인 좌표)을 궤적에 포함
        for b in BALLS:
            for f, x, y in self.hist[b]:
                if f >= frame_idx - STILL_WIN:
                    self.shot["traj"][b].append((f, x, y))
        self.state = "SHOT"

    def _finalize_shot(self, frame_idx, settled, layout=None):
        shot = self.shot
        self.shot = None
        if shot is None:
            return
        if settled:
            after = layout
            after_source = "settled"
        else:
            after = {}
            for b in BALLS:
                pts = shot["traj"][b] or [(0,) + shot["before"][b]]
                after[b] = (pts[-1][1], pts[-1][2])
            after_source = "last_seen"

        success, detail = self._judge_success(shot)
        self.turns.append({
            "epoch": self.epoch,
            "shooter": shot["cue"],
            "before": shot["before"],
            "after": after,
            "after_source": after_source,
            "success": success,
            "success_detail": detail,
            "frame_start": shot["frame_start"],
            "frame_end": frame_idx,
            "traj": shot["traj"],   # 판정 파라미터 튜닝용 원본 궤적 (traj.json으로 분리 저장)
        })

    # ---------- 성공 판정 (궤적 분석) ----------
    def _judge_success(self, shot):
        cue = shot["cue"]
        objs = [b for b in BALLS if b != cue]
        traj = {b: [(f, *to_m(x, y)) for f, x, y in shot["traj"][b]] for b in BALLS}
        cue_tr = traj[cue]
        n_obs = sum(1 for f, _, _ in cue_tr if f >= shot["frame_start"])
        n_frames = max(cue_tr[-1][0] - shot["frame_start"], 1) if cue_tr else 1
        coverage = min(n_obs / n_frames, 1.0)

        detail = {"method": "trajectory", "coverage": round(coverage, 2),
                  "hits": [], "cushions_before_2nd": None}
        if len(cue_tr) < 5 or coverage < COVERAGE_MIN:
            detail["method"] = "insufficient"
            return None, detail

        # 쿠션 접촉: 4개 레일 존 진입 횟수 (히스테리시스로 중복 방지).
        # 샷 시작 시 이미 존 안이면(레일에 붙여 놓고 치는 경우) 첫 진입은 세지 않는다.
        rails = (("x0", 0, 0), ("x1", 0, TABLE_W), ("y0", 1, 0), ("y1", 1, TABLE_H))
        cushion_frames = []
        inside = {}
        for i, (f, x, y) in enumerate(cue_tr):
            if i > 0:
                f0, x0, y0 = cue_tr[i - 1]
                speed = np.hypot(x - x0, y - y0) / max(f - f0, 1) * self.fps
            else:
                speed = 0.0
            for name, axis, edge in rails:
                dist = abs((x, y)[axis] - edge)
                if name not in inside:
                    inside[name] = dist < CUSHION_ZONE_M    # 초기 상태
                    continue
                if inside[name]:
                    if dist > CUSHION_EXIT_M:
                        inside[name] = False
                elif dist < CUSHION_ZONE_M and speed > CUSHION_V_MIN:
                    inside[name] = True
                    cushion_frames.append(f)

        # 목적구 접촉: 접촉하면 목적구가 반드시 움직인다 → "움직임 시작 시점에
        # 수구가 근처에 있었는가"로 판정. 단순 근접 통과(스침)는 세지 않는다.
        cue_by_f = {f: (x, y) for f, x, y in cue_tr}
        cue_x0, cue_y0 = to_m(*shot["before"][cue])
        hit_frame = {}
        for o in objs:
            ox0, oy0 = to_m(*shot["before"][o])
            move_pt = None      # 목적구가 처음 움직인 관측 (지속 이동 확인)
            for j, (f, x, y) in enumerate(traj[o]):
                if ((x - ox0) ** 2 + (y - oy0) ** 2) ** 0.5 > OBJ_MOVE_M:
                    nxt = traj[o][j + 1:j + 3]
                    if not nxt or any(((x2 - ox0) ** 2 + (y2 - oy0) ** 2) ** 0.5 > OBJ_MOVE_M
                                      for _, x2, y2 in nxt):
                        move_pt = (f, x, y)
                        break
            if move_pt is None:
                continue
            mf, mx, my = move_pt
            cands = [cue_by_f[g] for g in range(mf - 6, mf + 3) if g in cue_by_f]
            if not cands:
                continue
            d_cue = min(((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5 for cx, cy in cands)
            cue_departed = any(((cx - cue_x0) ** 2 + (cy - cue_y0) ** 2) ** 0.5 > 0.05
                               for cx, cy in cands)
            if d_cue < HIT_NEAR_M and cue_departed:
                hit_frame[o] = mf

        detail["hits"] = sorted(hit_frame, key=hit_frame.get)
        if len(hit_frame) < 2:
            return False, detail
        f_second = max(hit_frame.values())
        n_cushion = sum(1 for f in cushion_frames if f <= f_second + 1)
        detail["cushions_before_2nd"] = n_cushion
        return n_cushion >= 3, detail


def drop_replay_turns(turns, reader):
    """리플레이(다시보기) 구간에서 생긴 유령 턴 제거.
    방송은 리플레이 중 점수판을 숨기므로, 탑뷰 리플레이가 새 샷으로 오인돼도
    샷 구간의 점수판 노출 비율이 낮다 → 실제 샷이 아니라고 판정.
    (kept, dropped) 반환. dropped 는 (턴, 노출비율) 목록."""
    kept, dropped = [], []
    for t in turns:
        frac = reader.visible_fraction(t["frame_start"], t["frame_end"])
        if frac < REPLAY_VISIBLE_MAX:
            dropped.append((t, frac))
        else:
            kept.append(t)
    return kept, dropped


def apply_scoreboard_judgment(turns, reader, fps):
    """점수판 이벤트로 턴별 성공 여부를 재판정한다.
    턴 i의 창 = [frame_start_i, frame_start_{i+1}) — 득점 반영은 다음 샷 전에 이뤄진다.
    창 안에서 수구 색 점수가 +1/+2 오른 스텝이 있으면 성공.
    +2는 뱅크샷(3쿠션 선행 후 두 목적구 접촉 = 2점) → bank_shot=True 로 기록.
    스텝이 없으면 실패 — 단, 창 끝까지 판독이 이어졌을 때만 확정한다.
    점수가 이상하게 튄 창(편집 컷 등)은 건너뛰고 궤적 판정을 유지한다.
    적용한 턴 수를 반환."""
    col = {"white": 1, "yellow": 2}
    ev = reader.events
    steps = [(ev[i][0], ev[i][1] - ev[i - 1][1], ev[i][2] - ev[i - 1][2])
             for i in range(1, len(ev))]
    n_applied = 0
    for i, t in enumerate(turns):
        cue = t["shooter"]
        if cue not in col:
            continue
        start = t["frame_start"]
        end = turns[i + 1]["frame_start"] if i + 1 < len(turns) else None
        if reader.first_read_frame is None or reader.first_read_frame > start:
            continue                    # 창 시작 전에 기준 점수가 없음
        win = [s for s in steps
               if start <= s[0] < (end if end is not None else float("inf"))]
        me, opp = col[cue], 3 - col[cue]
        clean = [s for s in win if s[me] in (1, 2) and s[opp] == 0]
        reset = [s for s in win if s[1] <= 0 and s[2] <= 0]   # 세트 종료 리셋
        if any(s not in clean and s not in reset for s in win):
            continue                    # 점수가 튐(하이라이트 컷 등) → 궤적 판정 유지
        if clean:
            ok = True
        else:
            # '실패' 확정은 창 끝(마지막 턴은 샷 종료 + 여유)까지 판독됐어야 신뢰 가능
            need = end if end is not None else t["frame_end"] + SCORE_FAIL_MARGIN_S * fps
            if reader.last_read_frame is None or reader.last_read_frame < need:
                continue
            ok = False
        det = t["success_detail"]
        det["traj_success"] = t["success"]      # 궤적 판정 결과 보존 (QA 비교용)
        det["method"] = "scoreboard"
        det["score_steps"] = [[int(f), int(dw), int(dy)] for f, dw, dy in win]
        # 창에 스텝이 여럿이면 첫 스텝이 이 턴의 득점 (뒤 스텝은 놓친 턴의 것)
        det["bank_shot"] = bool(ok and clean[0][me] == 2)
        t["success"] = ok
        n_applied += 1
    return n_applied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-frames", action="store_true",
                        help="턴별 직전/이후 프레임을 qa/ 폴더에 저장")
    parser.add_argument("--no-scoreboard", action="store_true",
                        help="점수판 OCR 판정 끄기 (궤적 판정만 사용)")
    args = parser.parse_args()
    video_id = args.video_id or os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.outdir, exist_ok=True)
    qa_dir = os.path.join(args.outdir, "qa")
    if args.save_frames:
        os.makedirs(qa_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"영상을 못 읽음: {args.video}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"입력: {args.video} ({total}프레임, {fps:.1f}fps)")

    model = YOLO(MODEL_PATH)
    extractor = TurnExtractor(fps=fps / args.every * args.every)  # fps는 원본 기준
    score_reader = None
    if not args.no_scoreboard:
        score_reader = ScoreReader()
        if not score_reader.enabled:
            print("점수판 OCR 비활성: tesseract 미설치(brew install tesseract) → 궤적 판정만 사용")
            score_reader = None
    corners = None
    corner_cands = []       # 기준 잠금 전, 탑뷰 프레임들의 꼭짓점 모음
    n_proc = n_top = 0
    n_turns_saved = 0
    t0 = time.time()

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
        n_proc += 1

        # 점수판은 탑뷰 여부와 무관하게 샘플링 (클로즈업 화면에도 떠 있다)
        if score_reader and n_proc % SCORE_EVERY == 0:
            score_reader.sample(frame_idx, frame)

        fast = detect_corners_fast(frame)
        is_plausible = plausible_top_view(fast, frame.shape)
        if corners is None:
            # 기준 꼭짓점을 매 프레임 비교와 '같은 함수'(detect_corners_fast)로 잡는다.
            # 정밀 함수(find_table_corners)로 잡으면 축소본 비교값과 수십 px 어긋나
            # 대부분의 탑뷰 프레임이 컷으로 오판돼 버려진다(방송에 따라 발생).
            # 첫 한 프레임 대신 여러 프레임의 중앙값으로 잠가 이상치에도 견고하게.
            if is_plausible:
                corner_cands.append(fast)
                if len(corner_cands) >= CORNER_LOCK_N:
                    corners = np.median(np.array(corner_cands), axis=0).astype(np.float32)
                    print(f"당구대 꼭짓점 (프레임 {frame_idx}, {len(corner_cands)}프레임 합의):",
                          corners.astype(int).tolist())
            continue
        top_view = is_plausible and np.abs(fast - corners).max() < CORNER_CUT_TOL
        if not top_view:
            extractor.tick(frame_idx, [])
            continue
        n_top += 1

        res = model(frame, verbose=False)[0]
        balls = [(res.names[int(b.cls)], float(b.conf), *b.xywh[0][:2].tolist())
                 for b in res.boxes]
        if balls:
            tbl = px_to_table([[b[2], b[3]] for b in balls], corners)
            balls_tbl = [(n, c, float(tx), float(ty))
                         for (n, c, _, _), (tx, ty) in zip(balls, tbl)]
        else:
            balls_tbl = []
        extractor.tick(frame_idx, balls_tbl)

        # 샷 시작 시점 프레임 저장 (점수판 포함 → 성공 여부 수동 검증용)
        if args.save_frames and extractor.state == "SHOT" and extractor.shot \
                and extractor.shot["frame_start"] == frame_idx:
            cv2.imwrite(os.path.join(
                qa_dir, f"{video_id}_turn{len(extractor.turns) + 1:02d}_before.jpg"), frame)

        # 새 턴이 확정되면 QA 프레임 저장 + 진행 로그
        if len(extractor.turns) > n_turns_saved:
            t = extractor.turns[-1]
            n_turns_saved = len(extractor.turns)
            print(f"  턴 {n_turns_saved}: {t['shooter']} 수구, "
                  f"성공={t['success']} ({t['success_detail']}), "
                  f"프레임 {t['frame_start']}~{t['frame_end']}")
            if args.save_frames:
                cv2.imwrite(os.path.join(qa_dir, f"{video_id}_turn{n_turns_saved:02d}_after.jpg"),
                            frame)

    extractor.flush(frame_idx)
    cap.release()

    # 점수판 기반: ① 리플레이 유령 턴 제거 → ② 성공 재판정 (판독 실패 턴은 궤적 유지)
    if score_reader and score_reader.locked and score_reader.events:
        kept, dropped = drop_replay_turns(extractor.turns, score_reader)
        for t, frac in dropped:
            print(f"  리플레이 유령 턴 제거: 프레임 {t['frame_start']}~{t['frame_end']} "
                  f"(점수판 노출 {frac:.0%})")
        extractor.turns = kept
        n_sb = apply_scoreboard_judgment(extractor.turns, score_reader, fps)
        print(f"점수판 판정: {n_sb}/{len(extractor.turns)}턴 적용, 유령 턴 {len(dropped)}개 제거 "
              f"(점수 이벤트 {len(score_reader.events)}개, 나머지는 궤적 판정 유지)")
    elif score_reader:
        print("점수판 미검출 → 전 턴 궤적 판정 유지")

    turns = extractor.turns
    jsonl_path = os.path.join(args.outdir, "turns.jsonl")
    with open(jsonl_path, "w") as f:
        for i, t in enumerate(turns, 1):
            rec = {
                "video_id": video_id, "turn": i, "epoch": t["epoch"],
                "shooter": t["shooter"],
                "before": {b: [round(v, 4) for v in t["before"][b]] for b in BALLS},
                "after": {b: [round(v, 4) for v in t["after"][b]] for b in BALLS},
                "after_source": t["after_source"],
                "success": t["success"],
                "success_detail": t["success_detail"],
                "frame_start": t["frame_start"], "frame_end": t["frame_end"],
                "time_start_s": round(t["frame_start"] / fps, 2),
                "time_end_s": round(t["frame_end"] / fps, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 원본 궤적 (판정 파라미터 튜닝/디버깅용)
    with open(os.path.join(args.outdir, "traj.json"), "w") as f:
        json.dump([{ "turn": i, "frame_start": t["frame_start"], "shooter": t["shooter"],
                     "before": t["before"],
                     "traj": {b: [[fr, round(x, 4), round(y, 4)] for fr, x, y in t["traj"][b]]
                              for b in BALLS}}
                   for i, t in enumerate(turns, 1)], f)

    csv_path = os.path.join(args.outdir, "turns.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "turn", "shooter", "success", "after_source",
                    "time_start_s", "time_end_s"]
                   + [f"before_{b}_{a}" for b in BALLS for a in "xy"]
                   + [f"after_{b}_{a}" for b in BALLS for a in "xy"])
        for i, t in enumerate(turns, 1):
            w.writerow([video_id, i, t["shooter"], t["success"], t["after_source"],
                        round(t["frame_start"] / fps, 2), round(t["frame_end"] / fps, 2)]
                       + [round(v, 4) for b in BALLS for v in t["before"][b]]
                       + [round(v, 4) for b in BALLS for v in t["after"][b]])

    elapsed = time.time() - t0
    print(f"\n처리: {n_proc}프레임 in {elapsed:.1f}s ({n_proc / max(elapsed, 1e-9):.1f} fps), "
          f"탑뷰 {n_top}프레임")
    print(f"추출된 턴: {len(turns)}개 → {jsonl_path}, {csv_path}")


if __name__ == "__main__":
    main()
