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

# 스핀(당점)·커브는 1차 버전에서 미구현 → 실제보다 보수적인 확률이 나온다.


def simulate_shots(pos0, angles, speeds, record=False):
    """pos0: (3,2) 미터 좌표, 인덱스 0=수구.
    모든 (각도x힘) 샷을 numpy로 동시에 진행시킨다.
    반환: success(S,), ang(S,), spd(S,), cushions(S,), frames(기록 시)"""
    ang_grid, spd_grid = np.meshgrid(angles, speeds, indexing="ij")
    ang, spd = ang_grid.ravel(), spd_grid.ravel()
    S = ang.size

    pos = np.repeat(pos0[None], S, axis=0).astype(np.float64)  # (S,3,2)
    vel = np.zeros_like(pos)
    vel[:, 0, 0] = spd * np.cos(ang)
    vel[:, 0, 1] = spd * np.sin(ang)

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

        # ---------- 공-공 충돌 (등질량 탄성 충돌) ----------
        for i, j in ((0, 1), (0, 2), (1, 2)):
            diff = pos[:, j] - pos[:, i]
            dist = np.linalg.norm(diff, axis=1)
            n = diff / np.maximum(dist, 1e-9)[:, None]
            v_rel_n = np.einsum("sk,sk->s", vel[:, i] - vel[:, j], n)
            collide = active & (dist < 2 * BALL_R) & (v_rel_n > 0)
            if not collide.any():
                continue

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

        # ---------- 구름 마찰 감속 ----------
        speed = np.linalg.norm(vel, axis=2)
        moving = speed > 0
        scale = np.zeros_like(speed)
        scale[moving] = np.maximum(speed[moving] - DECEL * DT, 0) / speed[moving]
        vel *= scale[:, :, None]

        done |= (speed < STOP_V).all(axis=1)    # 모두 멈추면 실패로 종료

        if record and step % 10 == 0:
            frames.append(pos.copy())

    return success, ang, spd, cushions, frames


# ---------- 궤적 시각화 ----------
def draw_shot(pos0, angle, speed, names, out_path):
    _, _, _, _, frames = simulate_shots(pos0, np.array([angle]), np.array([speed]), record=True)
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


def estimate_probability(balls, cue="white", n_angles=360, speeds=(1.5, 2.5, 3.5)):
    """balls: {'white': (x,y), 'yellow': ..., 'red': ...} 0~1 정규화 좌표.
    반환: (확률, 성공 샷 목록 [(각도deg, 속도), ...], 공 이름 순서, 미터 좌표)"""
    names = [cue] + [n for n in ("white", "yellow", "red") if n != cue]
    pos0 = np.array([[balls[n][0] * TABLE_W, balls[n][1] * TABLE_H] for n in names])

    angles = np.deg2rad(np.arange(n_angles) * 360.0 / n_angles)
    success, ang, spd, _, _ = simulate_shots(pos0, angles, np.array(speeds))

    shots = [(float(np.rad2deg(a)), float(v)) for a, v in zip(ang[success], spd[success])]
    return float(success.mean()), shots, names, pos0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cue", choices=["white", "yellow"], default="white", help="수구 선택")
    args = parser.parse_args()

    # detect_pipeline.py 출력 예시 좌표 (test_frame.png 탐지 결과)
    balls = {"white": (0.443, 0.373), "red": (0.751, 0.350), "yellow": (0.408, 0.649)}

    prob, shots, names, pos0 = estimate_probability(balls, cue=args.cue)

    print(f"수구: {args.cue}, 공 배치: {balls}")
    print(f"\n3쿠션 성공 확률: {prob:.1%} ({len(shots)}/{360 * 3} 샷 성공)")

    by_speed = {}
    for a, v in shots:
        by_speed.setdefault(v, []).append(a)
    for v in sorted(by_speed):
        degs = ", ".join(f"{a:.0f}°" for a in sorted(by_speed[v])[:10])
        more = "" if len(by_speed[v]) <= 10 else f" 외 {len(by_speed[v]) - 10}개"
        print(f"  속도 {v} m/s: {len(by_speed[v])}개 성공 (각도: {degs}{more})")

    if shots:
        a, v = min(shots, key=lambda s: s[1])   # 가장 약한 힘으로 성공한 샷을 시각화
        draw_shot(pos0, np.deg2rad(a), v, names, "sim_shot.png")
        print(f"\n저장: sim_shot.png (각도 {a:.0f}°, 속도 {v} m/s 성공 샷 궤적)")


if __name__ == "__main__":
    main()
