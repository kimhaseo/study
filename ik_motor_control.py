"""
IK로 계산된 관절값을 모터에 전송.

사용법:
    python ik_motor_control.py
    또는 __main__ 블록에서 목표 위치/자세 직접 수정.
"""

import numpy as np
import pinocchio as pin

from ik_simple import solve_ik, euler_to_rotation, URDF_PATH, EE_FRAME
from motor_controller import MotorController
from config.motor_cmd import AngleCommand

# pinocchio q_deg 인덱스 → 모터 이름
# joint_1(idx=0): base (미사용), joint_8(idx=7): 그리퍼 등 (미사용)
JOINT_TO_MOTOR = {
    1: "left_joint1",
    2: "left_joint2",
    3: "left_joint3",
    4: "left_joint4",
    5: "left_joint5",
    6: "left_joint6",
}

MOTOR_SPEED = 360  # deg/s


def make_commands(q_deg: np.ndarray, speed: int = MOTOR_SPEED) -> list[AngleCommand]:
    """IK 결과(degree 배열)를 AngleCommand 리스트로 변환."""
    return [
        AngleCommand(motor_name, float(q_deg[idx]), speed)
        for idx, motor_name in JOINT_TO_MOTOR.items()
    ]


def run(target_pos: np.ndarray, target_rot: np.ndarray,
        model, data, ee_id) -> bool:
    """
    IK 풀고 모터에 전송.

    Returns:
        True  : IK 성공 + 전송 완료
        False : IK 실패
    """
    q_deg, pos_err, rot_err = solve_ik(model, data, ee_id, target_pos, target_rot)

    if q_deg is None:
        print(f"IK 실패  위치오차={pos_err*1000:.2f} mm  자세오차={np.degrees(rot_err):.3f}°")
        return False

    print(f"IK 성공  위치오차={pos_err*1000:.2f} mm  자세오차={np.degrees(rot_err):.3f}°")

    commands = make_commands(q_deg)
    print("전송할 관절값:")
    for cmd in commands:
        print(f"  {cmd.motor_name:12s} → {cmd.angle:8.3f}°  (CAN {hex(cmd.can_id)})")

    with MotorController() as mc:
        mc.move_motors(commands)
    print("모터 전송 완료.\n")
    return True


if __name__ == "__main__":
    # ── 목표 위치·자세 설정 ─────────────────────────────────
    target_pos = np.array([0.3, 0.0, 0.2])     # x, y, z (m)
    target_rot = euler_to_rotation(0, 0, 0)     # roll, pitch, yaw (deg)
    # ────────────────────────────────────────────────────────

    model = pin.buildModelFromUrdf(URDF_PATH)
    data  = model.createData()
    ee_id = model.getFrameId(EE_FRAME)

    print(f"목표 위치 : {target_pos}")
    print(f"목표 자세 : roll=0°  pitch=0°  yaw=0°\n")

    run(target_pos, target_rot, model, data, ee_id)
