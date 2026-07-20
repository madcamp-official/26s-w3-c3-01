# 방송 점수판 판독기: 화면 왼쪽 하단의 흰/노란 점수 박스를 찾아 숫자를 읽고,
# 값이 바뀐 시점을 (frame, 흰점수, 노란점수) 이벤트로 기록한다.
#
# 왜 점수판인가: 3쿠션 중계는 득점 시 오퍼레이터가 점수를 즉시 올리므로 사실상 정답지다.
# 궤적 분석(쿠션 수 세기)보다 정확하다. PBA 점수판은 박스 색 = 수구 색이라
# (흰 박스=흰 수구 선수, 노란 박스=노란 수구 선수) 수구와 바로 매칭된다.
#
# 의존: tesseract 바이너리(brew install tesseract) + pytesseract 패키지.
# 없으면 reader.enabled=False 가 되어 호출부(extract_turns)가 궤적 판정으로 폴백한다.
import cv2
import numpy as np

try:
    import pytesseract
    _HAVE_TESS = True
except ImportError:
    _HAVE_TESS = False

# 점수판 탐색 영역(화면 비율) — 방송 점수판은 왼쪽 하단에 고정된다
SEARCH_X_MAX = 0.5
SEARCH_Y_MIN = 0.55
# 노란 점수 박스 HSV 범위 / 흰 박스 판정 기준
YELLOW_LO = np.array([20, 110, 130])
YELLOW_HI = np.array([38, 255, 255])
WHITE_S_MAX = 60
WHITE_V_MIN = 170
BOX_AREA_MIN = 0.0008   # 박스 넓이 / 프레임 넓이 하한 (탁자 위 노란 공 배제)
BOX_AREA_MAX = 0.02
BOX_EXTENT_MIN = 0.85   # 윤곽 충전율 — 사각형만 통과 (원형 공/아이콘 ≈ 0.79 배제)
LOCK_N = 5              # 이 횟수 연속 같은 위치에서 찾으면 ROI 잠금
DIFF_TH = 3.5           # 점수 영역 평균 픽셀 변화가 이보다 크면 OCR 재실행
HEARTBEAT_SAMPLES = 20  # 변화가 없어도 이 샘플 수마다 한 번은 OCR (판독 시각 갱신용)
SCORE_MAX = 40          # 이 값 초과 판독은 오독으로 버림

# --- 현재 이닝(런) 원형 표시 ---
# 점수 박스 오른쪽의 흰/노란 원. 숫자가 떠 있는 쪽이 지금 치는 선수(= 수구 색)이고,
# 값은 이번 이닝 득점 누계(득점마다 +1, 뱅크샷 +2). 상대 원은 숫자 없이 비어 있다.
# 원에 0이 새로 나타나는 순간이 턴 교대(다음 선수 시작) 시점이다.
CIRCLE_SEARCH_X = 3.0        # 박스 오른쪽으로 박스 폭의 몇 배까지 원을 찾나
CIRCLE_EXTENT_MIN = 0.6      # 원형 충전율 범위 (원 ≈ 0.785, 사각형 ≈ 1.0 배제)
CIRCLE_EXTENT_MAX = 0.92
CIRCLE_DIGIT_DARK_MIN = 0.03 # 원 중앙부 어두운 픽셀 비율이 이보다 크면 숫자가 있음
CIRCLE_TRY_MAX = 600         # 이 샘플 수 안에 원을 못 찾으면 포기(레이아웃 다른 방송)
RUN_MAX = 30                 # 이닝 점수 상한 (초과 판독은 오독)


class ScoreReader:
    """sample(frame_idx, frame)을 주기적으로 호출하면 events 에
    (frame, 흰점수, 노란점수) 변화 이력이 쌓인다. 2회 연속 같은 판독만 확정해
    득점 플래시 애니메이션·오독을 걸러낸다."""

    def __init__(self):
        self.enabled = _HAVE_TESS and self._tess_ok()
        self.locked = False
        self.box_white = None       # (x, y, w, h) 전체 프레임 좌표
        self.box_yellow = None
        self.events = []            # [(frame, white, yellow)] 값이 바뀐(확정된) 시점
        self.first_read_frame = None
        self.last_read_frame = None
        # 점수판 가시성 타임라인: 상태가 바뀐 시점만 (frame, visible) 기록.
        # 방송은 리플레이 중 점수판을 숨기므로, 이걸로 리플레이 구간을 식별한다.
        self.vis_changes = []
        self._visible = None
        self._cands = []
        self._committed = None
        self._pending = None        # (frame, (w, y)) 확정 대기 중 판독
        self._sig = None            # 변화 감지용 축소 그레이 서명
        self._since_ocr = 0
        # 현재 이닝 원형 표시 (수구/턴 교대 판별)
        self.circle_white = None    # (x, y, w, h)
        self.circle_yellow = None
        self.circles_locked = False
        self.active_events = []     # [(frame, "white"|"yellow", 이닝점수)] 변화 시점
        self._circle_cands = []
        self._circle_tries = 0
        self._active_committed = None
        self._active_pending = None

    @staticmethod
    def _tess_ok():
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    # ---------- 점수 박스 위치 탐지 ----------
    def _try_locate(self, frame):
        """노란 사각 박스를 앵커로 찾고, 바로 위/아래에 같은 크기 흰 박스가
        붙어 있으면 점수판으로 인정. LOCK_N회 연속 검출되면 중앙값으로 잠금."""
        h, w = frame.shape[:2]
        y0 = int(h * SEARCH_Y_MIN)
        roi = frame[y0:h, 0:int(w * SEARCH_X_MAX)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = h * w
        found = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if not (BOX_AREA_MIN * frame_area < bw * bh < BOX_AREA_MAX * frame_area):
                continue
            if cv2.contourArea(c) / (bw * bh) < BOX_EXTENT_MIN:
                continue
            if not (0.5 < bw / bh < 2.2):
                continue
            for dy in (-bh, bh):        # 흰 박스는 노란 박스 바로 위 또는 아래
                wy = by + dy
                if wy < 0 or wy + bh > roi.shape[0]:
                    continue
                sub = hsv[wy:wy + bh, bx:bx + bw]
                white_frac = np.mean((sub[..., 1] < WHITE_S_MAX)
                                     & (sub[..., 2] > WHITE_V_MIN))
                if white_frac > 0.55:
                    found = ((bx, wy + y0, bw, bh), (bx, by + y0, bw, bh))
                    break
            if found:
                break
        if found is None:
            self._cands.clear()         # 연속 검출 요구 — 끊기면 처음부터
            return
        self._cands.append(found)
        if len(self._cands) < LOCK_N:
            return
        arr = np.array(self._cands)     # (N, 2, 4)
        med = np.median(arr, axis=0).astype(int)
        if np.abs(arr - med).max() > max(med[0, 2], med[0, 3]):
            self._cands = self._cands[-1:]      # 위치가 튐 — 최근 것만 남기고 재수집
            return
        self.box_white, self.box_yellow = tuple(med[0]), tuple(med[1])
        xs = [self.box_white[0], self.box_yellow[0]]
        ys = [self.box_white[1], self.box_yellow[1]]
        x2 = max(b[0] + b[2] for b in (self.box_white, self.box_yellow))
        y2 = max(b[1] + b[3] for b in (self.box_white, self.box_yellow))
        self._sig_box = (min(xs), min(ys), x2 - min(xs), y2 - min(ys))
        self.locked = True

    # ---------- 현재 이닝 원형 표시 (수구 판별) ----------
    def _find_circle_near(self, frame, box, kind):
        """점수 박스 오른쪽에서 같은 색 원형 표시를 찾는다. 실패 시 None."""
        x, y, w, h = box
        x1, x2 = x + w, min(frame.shape[1], x + w + int(CIRCLE_SEARCH_X * w))
        y1, y2 = max(0, y - h // 3), min(frame.shape[0], y + h + h // 3)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if kind == "white":
            mask = (((hsv[..., 1] < WHITE_S_MAX) & (hsv[..., 2] > WHITE_V_MIN))
                    .astype(np.uint8) * 255)
        else:
            mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            if not (0.5 * h < bh < 1.4 * h):        # 박스 높이와 비슷한 크기만
                continue
            if not (0.7 < bw / bh < 1.4):           # 대략 원형 비율
                continue
            ext = cv2.contourArea(c) / (bw * bh)
            if not (CIRCLE_EXTENT_MIN < ext < CIRCLE_EXTENT_MAX):
                continue                            # 사각형(박스)·가는 아이콘 배제
            if best is None or bx < best[0]:        # 박스에서 가장 가까운 후보
                best = (bx, by, bw, bh)
        if best is None:
            return None
        bx, by, bw, bh = best
        return (x1 + bx, y1 + by, bw, bh)

    def _try_locate_circles(self, frame):
        cw = self._find_circle_near(frame, self.box_white, "white")
        cy = self._find_circle_near(frame, self.box_yellow, "yellow")
        if cw is None or cy is None:
            self._circle_cands.clear()
            return
        self._circle_cands.append((cw, cy))
        if len(self._circle_cands) < LOCK_N:
            return
        arr = np.array(self._circle_cands)
        med = np.median(arr, axis=0).astype(int)
        if np.abs(arr - med).max() > max(med[0, 2], med[0, 3]):
            self._circle_cands = self._circle_cands[-1:]
            return
        self.circle_white, self.circle_yellow = tuple(med[0]), tuple(med[1])
        self.circles_locked = True
        # 변화 감지 서명 영역을 원까지 포함하도록 확장
        boxes = (self.box_white, self.box_yellow, self.circle_white, self.circle_yellow)
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes); y2 = max(b[1] + b[3] for b in boxes)
        self._sig_box = (x0, y0, x2 - x0, y2 - y0)
        self._sig = None                            # 서명 영역 바뀜 — 리셋

    def _read_circle(self, frame, box, kind):
        """원형 하나 판독 → (이닝점수 or None, 숫자유무, 가시성).
        빈 원 = (None, False, True) — 비활성 선수.
        숫자가 있는데 판독 실패 = (None, True, True)."""
        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if kind == "white":
            frac = np.mean((hsv[..., 1] < WHITE_S_MAX) & (hsv[..., 2] > WHITE_V_MIN))
        else:
            frac = np.mean(cv2.inRange(hsv, YELLOW_LO, YELLOW_HI) > 0)
        if frac < 0.35:
            return None, False, False               # 점수판이 화면에 없음
        m = max(2, int(0.18 * min(w, h)))           # 테두리 안티앨리어싱 배제
        inner = cv2.cvtColor(crop[m:h - m, m:w - m], cv2.COLOR_BGR2GRAY)
        if inner.size == 0 or float(np.mean(inner < 100)) < CIRCLE_DIGIT_DARK_MIN:
            return None, False, True                # 빈 원 — 비활성
        up = cv2.resize(inner, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.copyMakeBorder(th, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        for psm in (10, 8, 7):                      # 한 글자 → 단어 → 한 줄
            txt = pytesseract.image_to_string(
                th, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
            digits = "".join(ch for ch in txt if ch.isdigit())
            if digits:
                v = int(digits)
                return (v, True, True) if v <= RUN_MAX else (None, True, True)
        return None, True, True

    def _sample_circles(self, frame_idx, frame):
        """활성 선수(숫자 뜬 원)와 이닝 점수를 판독해 active_events 에 기록.
        점수와 같은 2회 연속 확정 방식."""
        rw, dw, vw = self._read_circle(frame, self.circle_white, "white")
        ry, dy, vy = self._read_circle(frame, self.circle_yellow, "yellow")
        if not (vw and vy):
            self._active_pending = None
            return
        if dw and not dy:
            color, run = "white", rw
        elif dy and not dw:
            color, run = "yellow", ry
        else:
            return                                  # 둘 다/둘 다 아님 → 모호, 보류
        if run is None:
            return                                  # 숫자 판독 실패 — 다음 샘플에서
        reading = (color, run)
        if reading == self._active_committed:
            self._active_pending = None
        elif self._active_pending and self._active_pending[1] == reading:
            self.active_events.append((self._active_pending[0], color, run))
            self._active_committed = reading
            self._active_pending = None
        else:
            self._active_pending = (frame_idx, reading)

    # ---------- 수구/이닝 조회 ----------
    def active_color_at(self, f0, f1):
        """샷 창 [f0, f1)의 수구 색 — f0 이전 마지막 활성 이벤트의 색.
        없으면 창 안 첫 이벤트의 색, 그래도 없으면 None."""
        last = None
        for f, color, _ in self.active_events:
            if f <= f0:
                last = color
            elif f < f1:
                if last is None:
                    last = color
                break
            else:
                break
        return last

    def run_steps_in(self, f0, f1, color):
        """창 안에서 해당 색 이닝 점수가 오른 스텝 [(frame, 증가량)] —
        같은 색 연속 이벤트끼리만 비교(턴 교대로 0 리셋은 스텝이 아님)."""
        steps, prev = [], None
        for f, c, run in self.active_events:
            if c != color:
                prev = None
                continue
            if prev is not None and f0 <= f < f1 and run > prev:
                steps.append((f, run - prev))
            prev = run
        return steps

    # ---------- 숫자 판독 ----------
    def _read_box(self, frame, box, kind):
        """박스 하나에서 (점수, 가시성)을 읽는다.
        가시성 False = 박스가 제 색이 아님 → 점수판이 화면에 없음(리플레이 등).
        점수 None + 가시성 True = 그래픽은 있는데 숫자 판독 실패."""
        x, y, w, h = box
        crop = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if kind == "white":
            frac = np.mean((hsv[..., 1] < WHITE_S_MAX) & (hsv[..., 2] > WHITE_V_MIN))
        else:
            frac = np.mean(cv2.inRange(hsv, YELLOW_LO, YELLOW_HI) > 0)
        if frac < 0.45:
            return None, False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.copyMakeBorder(th, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        for psm in (8, 7, 13):          # 단어 → 한 줄 → raw 순서로 시도
            txt = pytesseract.image_to_string(
                th, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789")
            digits = "".join(ch for ch in txt if ch.isdigit())
            if digits:
                v = int(digits)
                return (v, True) if v <= SCORE_MAX else (None, True)
        return None, True

    def visible_fraction(self, f0, f1):
        """[f0, f1] 프레임 구간에서 점수판이 화면에 있던 시간 비율.
        기록이 없으면 1.0(중립 — 필터가 오작동으로 턴을 지우지 않게)."""
        if f1 <= f0 or not self.vis_changes:
            return 1.0
        # f0 시점의 상태 = f0 이전 마지막 기록 (없으면 첫 기록 상태를 소급 적용)
        state = self.vis_changes[0][1]
        prev = f0
        total = vis = 0
        for f, s in self.vis_changes:
            if f <= f0:
                state = s
                continue
            if f >= f1:
                break
            total += f - prev
            vis += (f - prev) if state else 0
            prev, state = f, s
        total += f1 - prev
        vis += (f1 - prev) if state else 0
        return vis / total if total else 1.0

    # ---------- 주기 샘플링 ----------
    def sample(self, frame_idx, frame):
        if not self.enabled:
            return
        if not self.locked:
            self._try_locate(frame)
            if not self.locked:
                return
        # 박스 잠금 후: 오른쪽의 이닝 원형 표시도 잠금 시도 (없는 레이아웃이면 포기)
        if not self.circles_locked and self._circle_tries < CIRCLE_TRY_MAX:
            self._circle_tries += 1
            self._try_locate_circles(frame)
        # OCR은 느리므로 점수 영역 픽셀이 변한 순간에만 실행 (+주기적 하트비트,
        # 확정 대기 중이면 다음 샘플에서 바로 재판독)
        x, y, w, h = self._sig_box
        sig = cv2.resize(cv2.cvtColor(frame[y:y + h, x:x + w],
                                      cv2.COLOR_BGR2GRAY), (48, 32))
        self._since_ocr += 1
        changed = (self._sig is None
                   or float(np.mean(cv2.absdiff(sig, self._sig))) > DIFF_TH)
        if not (changed or self._pending or self._active_pending
                or self._since_ocr >= HEARTBEAT_SAMPLES):
            return
        self._sig = sig
        self._since_ocr = 0
        if self.circles_locked:
            self._sample_circles(frame_idx, frame)
        sw, vis_w = self._read_box(frame, self.box_white, "white")
        sy, vis_y = self._read_box(frame, self.box_yellow, "yellow")
        visible = vis_w and vis_y
        if visible != self._visible:
            self._visible = visible
            self.vis_changes.append((frame_idx, visible))
        if sw is None or sy is None:
            self._pending = None
            return
        reading = (sw, sy)
        if reading == self._committed:
            self.last_read_frame = frame_idx
            self._pending = None
        elif self._pending and self._pending[1] == reading:
            # 2회 연속 동일 → 확정. 이벤트 시각은 처음 관측된 프레임으로
            self.events.append((self._pending[0], sw, sy))
            self._committed = reading
            if self.first_read_frame is None:
                self.first_read_frame = self._pending[0]
            self.last_read_frame = frame_idx
            self._pending = None
        else:
            self._pending = (frame_idx, reading)
