"""
URDF에서 IK 계산 후 각 관절값을 degree로 출력.

사용법:
    python ik_simple.py
    또는 코드 하단 __main__ 블록에서 목표 위치/자세 직접 수정.
"""

import time
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

URDF_PATH = "/Users/kimhaseo/workspace/study/7dof_urdf/7dof_urdf.urdf"
EE_FRAME  = "end_effector-v1"


def _ik_single(model, data, ee_id, q0, p_target, R_target,
               n_steps, pos_tol, rot_tol, damping, dq_max):
    q = q0.copy()
    for _ in range(n_steps):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        oMf    = data.oMf[ee_id]
        pe_vec = p_target - oMf.translation
        re_vec = pin.log3(R_target @ oMf.rotation.T)

        pe = float(np.linalg.norm(pe_vec))
        re = float(np.linalg.norm(re_vec))

        if pe < pos_tol and re < rot_tol:
            return q, pe, re

        J  = pin.computeFrameJacobian(
            model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        JT = J.T
        A  = J @ JT + damping * np.eye(6)
        dq = JT @ np.linalg.solve(A, np.hstack([pe_vec, re_vec]))
        q  = pin.integrate(model, q, np.clip(dq, -dq_max, dq_max))
        # 관절 한계 적용
        q  = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[ee_id]
    pe  = float(np.linalg.norm(p_target - oMf.translation))
    re  = float(np.linalg.norm(pin.log3(R_target @ oMf.rotation.T)))
    return q, pe, re


def solve_ik(
    model,                      # pin.Model (미리 로딩된 모델)
    data,                       # pin.Data
    ee_id:      int,            # 엔드이펙터 프레임 ID
    p_target: np.ndarray,       # 목표 위치 [x, y, z] (meter)
    R_target: np.ndarray,       # 목표 자세 (3x3 rotation matrix)
    n_steps:    int   = 500,
    n_restarts: int   = 15,     # 실패 시 랜덤 재시작 횟수
    pos_tol:    float = 1e-3,   # 위치 허용 오차 (m)
    rot_tol:    float = 1e-2,   # 자세 허용 오차 (rad)
    damping:    float = 1e-4,
    dq_max:     float = 0.05,
) -> tuple[np.ndarray | None, float, float]:
    """
    Returns:
        q_joints (ndarray | None): 관절값 배열 (degree), 실패 시 None
        pos_err (float): 최종 위치 오차 (m)
        rot_err (float): 최종 자세 오차 (rad)
    """
    best_q, best_pe, best_re = None, np.inf, np.inf

    # neutral 시작 + 랜덤 재시작
    starts = [pin.neutral(model)] + [
        pin.randomConfiguration(model) for _ in range(n_restarts)
    ]

    for q0 in starts:
        q, pe, re = _ik_single(
            model, data, ee_id, q0, p_target, R_target,
            n_steps, pos_tol, rot_tol, damping, dq_max
        )
        if pe < pos_tol and re < rot_tol:
            return np.degrees(q), pe, re
        if pe + re < best_pe + best_re:
            best_q, best_pe, best_re = q, pe, re

    return None, best_pe, best_re


def euler_to_rotation(roll_deg, pitch_deg, yaw_deg) -> np.ndarray:
    """Roll/Pitch/Yaw (degree) → 3x3 rotation matrix (XYZ 순서)."""
    return R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


if __name__ == "__main__":
    # ── 반복 횟수 설정 ───────────────────────────────────────
    N_ITER = 10
    # ── 목표 위치·자세 설정 ──────────────────────────────────
    target_pos = np.array([0.3, 0.0, 0.2])          # x, y, z (m)
    target_rot = euler_to_rotation(0, 0, 0)          # roll, pitch, yaw (deg)
    # ────────────────────────────────────────────────────────

    model = pin.buildModelFromUrdf(URDF_PATH)
    data  = model.createData()
    ee_id = model.getFrameId(EE_FRAME)
    joint_names = [model.names[i] for i in range(1, model.njoints)]

    print(f"목표 위치 : {target_pos}")
    print(f"목표 자세 : roll=0°  pitch=0°  yaw=0°")
    print(f"반복 횟수 : {N_ITER}\n")

    elapsed_list = []
    for i in range(N_ITER):
        t0 = time.perf_counter()
        q_deg, pos_err, rot_err = solve_ik(model, data, ee_id, target_pos, target_rot)
        elapsed = (time.perf_counter() - t0) * 1000  # ms

        elapsed_list.append(elapsed)
        status = "성공" if q_deg is not None else "실패"
        print(f"[{i+1:2d}/{N_ITER}] {status}  |  {elapsed:7.1f} ms  |  "
              f"위치오차 {pos_err*1000:.2f} mm  |  자세오차 {np.degrees(rot_err):.3f}°")

    print(f"\n--- 지연시간 통계 ({N_ITER}회) ---")
    print(f"  최소 : {min(elapsed_list):.1f} ms")
    print(f"  최대 : {max(elapsed_list):.1f} ms")
    print(f"  평균 : {sum(elapsed_list)/len(elapsed_list):.1f} ms")
