# 3쿠션 물리 시뮬레이션 엔진
# 공 3개 좌표(0~1 정규화)를 받아 각도x힘 조합 샷을 병렬 시뮬레이션하고
# "3쿠션 성공 샷 비율"을 확률로 계산한다.
# 사용법: venv/bin/python simulate.py [--cue white|yellow]
import argparse
import cv2
import numpy as np

# ---------- 물리 상수 (국제식 대대 기준, 미터 단위) ----------
TABLE_W, TABLE_H = 2.84, 1.42   # 쿠션 안쪽 경기 면적
BALL_R = 0.0615 / 2             # 공 반지름 (지름 61.5mm)
DECEL = 0.12                    # 구름 마찰 감속 (m/s^2)
E_CUSHION = 0.85                # 쿠션 반발계수
E_BALL = 0.95                   # 공-공 반발계수
DT = 0.002                      # 시뮬레이션 시간 간격 (s)
MAX_T = 30.0                    # 샷 하나당 최대 시뮬레이션 시간
STOP_V = 0.02                   # 이 속도 미만이면 정지로 간주

# ---------- 스핀(당점) 상수 ----------
SIDE_FACTOR = 0.5               # 사이드 당점 → 속도 등가 스핀: spin = side * SIDE_FACTOR * 샷속도
K_SIDE = 0.55                   # 쿠션 반발 시 스핀이 접선 속도에 주는 영향
SPIN_CUSHION_DECAY = 0.6        # 쿠션 1회당 스핀 잔존율
SPIN_TIME_DECAY = 0.3           # 초당 스핀 자연 감쇠율
FOLLOW_COEF = 0.6               # 상/하 당점(밀어치기/끌어치기) 세기

# 미구현: 커브(마세), 스핀의 공-공 전달. 단순화 모델이므로 절대값보다
# 배치 간 비교 지표로 쓰는 것이 정확하다.


def simulate_shots(pos0, ang, spd, side=None, vert=None, record=False):
    """pos0: (3,2) 미터 좌표, 인덱스 0=수구. ang/spd/side/vert: (S,) 샷 파라미터.
    side(좌우 당점)·vert(상하 당점) ∈ [-1,1]. 사이드스핀은 쿠션 반발 각을,
    상/하 당점은 첫 목적구 충돌 직후 수구의 전진(팔로우)/후진(드로우)을 바꾼다.
    모든 샷을 numpy로 동시에 진행시킨다. 반환: success(S,), frames(기록 시)"""
    S = ang.size
    side = np.zeros(S) if side is None else side
    vert = np.zeros(S) if vert is None else vert

    pos = np.repeat(pos0[None], S, axis=0).astype(np.float64)  # (S,3,2)
    vel = np.zeros_like(pos)
    vel[:, 0, 0] = spd * np.cos(ang)
    vel[:, 0, 1] = spd * np.sin(ang)
    spin = side * SIDE_FACTOR * spd          # 수구 사이드스핀 (속도 등가, m/s)
    follow = vert.astype(np.float64).copy()  # 첫 목적구 충돌에서 소모

    cushions = np.zeros(S, np.int32)      # 수구의 쿠션 접촉 횟수
    hit = np.zeros((S, 2), bool)          # 목적구 1,2 접촉 여부
    done = np.zeros(S, bool)
    success = np.zeros(S, bool)
    frames = [] if record else None

    for step in range(int(MAX_T / DT)):
        active = ~done
        if not active.any():
            break

        pos[active] += vel[active] * DT

        # ---------- 쿠션 반사 ----------
        for axis, limit in ((0, TABLE_W), (1, TABLE_H)):
            low = active[:, None] & (pos[:, :, axis] < BALL_R) & (vel[:, :, axis] < 0)
            high = active[:, None] & (pos[:, :, axis] > limit - BALL_R) & (vel[:, :, axis] > 0)
            pos[:, :, axis][low] = 2 * BALL_R - pos[:, :, axis][low]
            pos[:, :, axis][high] = 2 * (limit - BALL_R) - pos[:, :, axis][high]
            bounce = low | high
            vel[:, :, axis][bounce] *= -E_CUSHION
            cushions[bounce[:, 0]] += 1   # 수구(인덱스 0)의 접촉만 카운트

            # 사이드스핀: 쿠션 접선(τ = ẑ×n) 방향으로 속도 성분 추가 후 스핀 감쇠
            other = 1 - axis
            tau_low = 1.0 if axis == 0 else -1.0
            cue_low, cue_high = low[:, 0], high[:, 0]
            vel[cue_low, 0, other] += K_SIDE * spin[cue_low] * tau_low
            vel[cue_high, 0, other] -= K_SIDE * spin[cue_high] * tau_low
            spin[cue_low | cue_high] *= SPIN_CUSHION_DECAY

        # ---------- 공-공 충돌 (등질량 탄성 충돌) ----------
        for i, j in ((0, 1), (0, 2), (1, 2)):
            diff = pos[:, j] - pos[:, i]
            dist = np.linalg.norm(diff, axis=1)
            n = diff / np.maximum(dist, 1e-9)[:, None]
            v_rel_n = np.einsum("sk,sk->s", vel[:, i] - vel[:, j], n)
            collide = active & (dist < 2 * BALL_R) & (v_rel_n > 0)
            if not collide.any():
                continue

            v_pre = vel[:, i].copy() if i == 0 else None  # 당점 효과용 충돌 직전 속도
            impulse = ((1 + E_BALL) / 2 * v_rel_n)[collide, None] * n[collide]
            vel[collide, i] -= impulse
            vel[collide, j] += impulse
            push = ((2 * BALL_R - dist) / 2)[collide, None] * n[collide]
            pos[collide, i] -= push
            pos[collide, j] += push

            # ---------- 3쿠션 룰 판정 (수구-목적구 접촉만 해당) ----------
            if i == 0:
                b = j - 1                        # 이번에 맞은 목적구 인덱스
                first_contact = collide & ~hit[:, b]
                second_ball = first_contact & hit[:, 1 - b]  # 남은 목적구를 맞춘 순간
                success |= second_ball & (cushions >= 3)
                done |= second_ball             # 득점/실패 여부가 결정됨
                hit[first_contact, b] = True

                # 상/하 당점: 목적구 충돌 직후 충돌 전 진행 방향으로 전진/후진
                spin_use = collide & (follow != 0)
                vel[spin_use, 0] += (FOLLOW_COEF * follow[spin_use])[:, None] * v_pre[spin_use]
                follow[spin_use] = 0.0

        # ---------- 구름 마찰 감속 + 스핀 자연 감쇠 ----------
        speed = np.linalg.norm(vel, axis=2)
        moving = speed > 0
        scale = np.zeros_like(speed)
        scale[moving] = np.maximum(speed[moving] - DECEL * DT, 0) / speed[moving]
        vel *= scale[:, :, None]
        spin *= 1 - SPIN_TIME_DECAY * DT

        done |= (speed < STOP_V).all(axis=1)    # 모두 멈추면 실패로 종료

        if record and step % 10 == 0:
            frames.append(pos.copy())

    return success, frames


# ---------- 궤적 시각화 ----------
def draw_shot(pos0, angle, speed, side, vert, names, out_path):
    _, frames = simulate_shots(pos0, np.array([angle]), np.array([speed]),
                               np.array([side]), np.array([vert]), record=True)
    traj = np.array([f[0] for f in frames])     # (T,3,2)

    scale = 300  # 1m = 300px
    img = np.full((int(TABLE_H * scale), int(TABLE_W * scale), 3), (140, 90, 30), np.uint8)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 255, 255), 3)

    bgr = {"white": (255, 255, 255), "yellow": (0, 180, 255), "red": (0, 0, 220)}
    to_px = lambda p: (int(p[0] * scale), int(p[1] * scale))
    for k, name in enumerate(names):
        for t in range(1, len(traj)):
            cv2.line(img, to_px(traj[t - 1, k]), to_px(traj[t, k]), bgr[name], 2)
        cv2.circle(img, to_px(traj[0, k]), int(BALL_R * scale), bgr[name], -1)
        cv2.circle(img, to_px(traj[0, k]), int(BALL_R * scale), (30, 30, 30), 2)
    cv2.imwrite(out_path, img)


def estimate_probability(balls, cue="white", n_angles=360, speeds=(1.5, 2.5, 3.5),
                         sides=(-0.7, 0.0, 0.7), verts=(-0.5, 0.0, 0.5),
                         aim_deg=None, aim_tol_deg=10.0):
    """balls: {'white': (x,y), 'yellow': ..., 'red': ...} 0~1 정규화 좌표.
    aim_deg를 주면 전 방향 대신 그 방향 ±aim_tol_deg 범위만 촘촘히 샘플링해
    '정해진 큐 방향으로 쳤을 때'의 조건부 확률을 계산한다.
    반환: (확률, 성공 샷 [(각도deg, 속도, side, vert), ...], 공 이름 순서, 미터 좌표)"""
    names = [cue] + [n for n in ("white", "yellow", "red") if n != cue]
    pos0 = np.array([[balls[n][0] * TABLE_W, balls[n][1] * TABLE_H] for n in names])

    if aim_deg is None:
        base = np.deg2rad(np.arange(n_angles) * 360.0 / n_angles)
    else:
        base = np.deg2rad(np.linspace(aim_deg - aim_tol_deg, aim_deg + aim_tol_deg, 41))
    grids = np.meshgrid(base, np.array(speeds), np.array(sides), np.array(verts), indexing="ij")
    ang, spd, side, vert = (g.ravel() for g in grids)

    success, _ = simulate_shots(pos0, ang, spd, side, vert)
    shots = [(float(np.rad2deg(a)), float(v), float(s), float(t))
             for a, v, s, t in zip(ang[success], spd[success], side[success], vert[success])]
    return float(success.mean()), shots, names, pos0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue", choices=["white", "yellow"], default="white", help="수구 선택")
    parser.add_argument("--aim", type=float, default=None,
                        help="큐 방향(도). 0°=오른쪽, 90°=아래(화면 기준). 주면 그 방향의 조건부 확률 계산")
    parser.add_argument("--aim-tol", type=float, default=10.0, help="큐 방향 허용 오차(±도)")
    args = parser.parse_args()

    # detect_pipeline.py 출력 예시 좌표 (test_frame.png 탐지 결과)
    balls = {"white": (0.443, 0.373), "red": (0.751, 0.350), "yellow": (0.408, 0.649)}

    prob, shots, names, pos0 = estimate_probability(balls, cue=args.cue)
    n_total = 360 * 3 * 3 * 3
    print(f"수구: {args.cue}, 공 배치: {balls}")
    print(f"\n[전 방향] 3쿠션 성공 확률: {prob:.1%} ({len(shots)}/{n_total} 샷, 스핀 포함)")
    no_spin = [s for s in shots if s[2] == 0 and s[3] == 0]
    print(f"  무회전 샷만: {len(no_spin)}/{360 * 3} ({len(no_spin) / (360 * 3):.1%}) → 스핀 효과 확인용")

    best = shots
    if args.aim is not None:
        prob_aim, shots_aim, _, _ = estimate_probability(
            balls, cue=args.cue, aim_deg=args.aim, aim_tol_deg=args.aim_tol)
        n_aim = 41 * 3 * 3 * 3
        print(f"\n[큐 방향 {args.aim:.0f}° ±{args.aim_tol:.0f}°] 조건부 성공 확률: "
              f"{prob_aim:.1%} ({len(shots_aim)}/{n_aim} 샷)")
        for a, v, s, t in sorted(shots_aim, key=lambda x: x[1])[:5]:
            print(f"  성공 예: 각도 {a:.1f}°, 속도 {v} m/s, 사이드 {s:+.1f}, 상하 {t:+.1f}")
        best = shots_aim

    if best:
        a, v, s, t = min(best, key=lambda x: x[1])  # 가장 약한 힘으로 성공한 샷을 시각화
        draw_shot(pos0, np.deg2rad(a), v, s, t, names, "sim_shot.png")
        print(f"\n저장: sim_shot.png (각도 {a:.0f}°, 속도 {v} m/s, 당점 사이드 {s:+.1f}/상하 {t:+.1f})")


if __name__ == "__main__":
    main()
