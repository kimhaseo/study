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

# =============================
# Initial Joint State
# =============================
q = pin.neutral(model)

# =============================
# ===== INPUT TARGET (DEG) =====
# =============================
p_des = np.array([-0.2, 0.1, -0.3])
R_des = ee_orientation_from_euler(0, 180, 0)

# =============================
# IK Parameters
# =============================
damping = 1e-3
dq_max = 0.05

# =============================
# IK Solve
# =============================
for i in range(300):
    pin.forwardKinematics(model, data, q)
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
        model, data, q, ee_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    J = J6[:, 6:]

    JT = J.T
    dq = JT @ np.linalg.inv(J @ JT + damping * np.eye(6)) @ err
    dq = np.clip(dq, -dq_max, dq_max)

    q[7:] += dq
    # q[7:] = 0.0  # 모든 관절을 0도로 초기화

# =============================
# OUTPUT IK RESULTS (DEGREE)
# =============================
print("\nSolved Joint Values (Degree):")
joint_angles_deg = []
for i, val in enumerate(q[7:]):
    deg_val = np.degrees(val)
    joint_angles_deg.append(deg_val)
    print(f"Joint {i+1}: {deg_val:.3f} deg")

# =============================
# VISUALIZE
# =============================
viz.display(q)
print("\nMeshcat viewer is open.")

# =============================
# MOTOR COMMAND 전송
# =============================
mc = MotorController()

# 예시 모터 이름 (left_joint1~6)
motor_names = [
    "left_joint1", "left_joint2", "left_joint3",
    "left_joint4", "left_joint5", "left_joint6"
]

# AngleCommand 리스트 생성
angle_commands = []
for name, angle in zip(motor_names, joint_angles_deg):
    # CAN ID 예시: 0x141~0x146 (환경에 맞게 수정)
    cmd = AngleCommand(motor_name=name, angle=angle)
    angle_commands.append(cmd)

# 모터로 명령 전송
mc.move_motors(angle_commands)
print("\nMotor commands sent based on IK solution.")

# =============================
# Keep Meshcat Open
# =============================
print("Close with Ctrl+C")
while True:
    time.sleep(1)
