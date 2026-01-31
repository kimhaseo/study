import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R
import time

# MotorController import
from motor_controller import MotorController
from config.motor_cmd import AngleCommand

# =============================
# URDF Path
# =============================
URDF_PATH = r"/Users/kimhaseo/workspace/study/test_arm/test_arm.urdf"
MESH_ROOT = r"/Users/kimhaseo/workspace/study/test_arm"

# =============================
# Pinocchio Model & Visual
# =============================
model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()

visual_model = pin.buildGeomFromUrdf(
    model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT]
)

viz = MeshcatVisualizer(model, None, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

EE_NAME = "link6"
ee_id = model.getFrameId(EE_NAME)

# =============================
# Utils
# =============================
def ee_orientation_from_euler(roll, pitch, yaw, degrees=True):
    return R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees).as_matrix()

def random_target_position():
    # 예시: x,y,z를 -0.3~0.3 범위에서 랜덤 선택
    x = np.random.uniform(-0.3, 0.3)
    y = np.random.uniform(0.05, 0.3)
    z = np.random.uniform(-0.3, -0.1)
    return np.array([x, y, z])

# =============================
# Initial Joint State
# =============================
q = pin.neutral(model)

# =============================
# MOTOR SETUP
# =============================
mc = MotorController()
motor_names = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6"
]
current_angles = np.array([0.0]*6)

# =============================
# 보간 파라미터
# =============================
duration = 2.0  # 목표까지 이동 시간 [s]
dt = 0.05       # 모터 업데이트 간격 [s]
steps = int(duration / dt)

print("Starting random target movements. Close with Ctrl+C")

while True:
    # =============================
    # 랜덤 목표 생성
    # =============================
    p_des = random_target_position()
    R_des = ee_orientation_from_euler(0, 180, 0)  # 자세는 고정

    # =============================
    # IK Solve
    # =============================
    q_ik = q.copy()
    for i in range(300):
        pin.forwardKinematics(model, data, q_ik)
        pin.updateFramePlacements(model, data)

        oMf = data.oMf[ee_id]
        p_cur = oMf.translation
        R_cur = oMf.rotation

        pos_err = p_des - p_cur
        rot_err = pin.log3(R_des @ R_cur.T)
        err = np.hstack([pos_err, rot_err])

        if np.linalg.norm(err) < 1e-5:
            break

        J6 = pin.computeFrameJacobian(
            model, data, q_ik, ee_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J6[:, 6:]

        JT = J.T
        dq = JT @ np.linalg.inv(J @ JT + 1e-3 * np.eye(6)) @ err
        dq = np.clip(dq, -0.05, 0.05)

        q_ik[7:] += dq

    # IK 결과를 degree로 변환
    target_angles = np.degrees(q_ik[7:])
    print(f"\nNew Target Position: {p_des}, Target Angles: {target_angles}")

    # =============================
    # 보간 이동 (모터 + Meshcat)
    # =============================
    for t in range(1, steps + 1):
        alpha = t / steps
        angles_step = (1 - alpha) * current_angles + alpha * target_angles

        # 모터 명령
        angle_commands = []
        for name, angle in zip(motor_names, angles_step):
            cmd = AngleCommand(motor_name=name, angle=angle)
            angle_commands.append(cmd)
        mc.move_motors(angle_commands)

        # Meshcat 시각화
        q_display = q.copy()
        q_display[7:] = np.radians(angles_step)
        viz.display(q_display)

        time.sleep(dt)

    # =============================
    # 최종 목표 각도로 맞춤
    # =============================
    current_angles = target_angles.copy()
