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
        # OCR은 느리므로 점수 영역 픽셀이 변한 순간에만 실행 (+주기적 하트비트,
        # 확정 대기 중이면 다음 샘플에서 바로 재판독)
        x, y, w, h = self._sig_box
        sig = cv2.resize(cv2.cvtColor(frame[y:y + h, x:x + w],
                                      cv2.COLOR_BGR2GRAY), (48, 32))
        self._since_ocr += 1
        changed = (self._sig is None
                   or float(np.mean(cv2.absdiff(sig, self._sig))) > DIFF_TH)
        if not (changed or self._pending or self._since_ocr >= HEARTBEAT_SAMPLES):
            return
        self._sig = sig
        self._since_ocr = 0
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
