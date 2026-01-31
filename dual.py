# -*- coding: utf-8 -*-
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R
import keyboard

URDF_PATH = r"C:\Users\kimha\PycharmProjects\study\test_arm\test_arm.urdf"
MESH_ROOT = r"C:\Users\kimha\PycharmProjects\study\test_arm"

# =========================================================
# Utils
# =========================================================
def ee_orientation_from_euler(roll, pitch, yaw, degrees=True):
    return R.from_euler(
        'xyz',
        [roll, pitch, yaw],
        degrees=degrees
    ).as_matrix()

def step_ik(model, data, q, ee_id, p_des, R_des, damping=1e-3):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    oMf = data.oMf[ee_id]
    p_cur = oMf.translation
    R_cur = oMf.rotation

    pos_err = p_des - p_cur
    rot_err = pin.log3(R_des @ R_cur.T)

    err = np.hstack([pos_err, rot_err])

    J6 = pin.computeFrameJacobian(
        model, data, q,
        ee_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    J = J6[:, 6:]   # Free-flyer(0~5) 제외한 조인트 공간

    dq = J.T @ np.linalg.inv(
        J @ J.T + damping * np.eye(6)
    ) @ err

    q[7:] += dq
    return q

# =========================================================
# Mirroring Helper
# =========================================================
# YZ 평면 대칭 행렬 (X축 반전)
M_mirror = np.diag([-1, 1, 1])

# =========================================================
# Models (LEFT / RIGHT)
# =========================================================
model_L = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
model_R = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())

data_L = model_L.createData()
data_R = model_R.createData()

visual_model_L = pin.buildGeomFromUrdf(
    model_L, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT]
)
visual_model_R = pin.buildGeomFromUrdf(
    model_R, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT]
)

viz = MeshcatVisualizer(model_L, None, visual_model_L)
viz.initViewer(open=True)
viz.loadViewerModel(rootNodeName="robot_L")

viz_R = MeshcatVisualizer(model_R, None, visual_model_R)
viz_R.viewer = viz.viewer
viz_R.loadViewerModel(rootNodeName="robot_R")

# =========================================================
# Initial q (거울 대칭 배치)
# =========================================================
q_L = pin.neutral(model_L)
q_R = pin.neutral(model_R)

# LEFT 로봇 위치 및 방향
q_L[0] = 0   # X축 왼쪽
q_L[1] = 0.2
q_L[2] = 0.0
q_L[3:7] = R.identity().as_quat()

# RIGHT 로봇: 위치는 X 대칭, 방향은 Z축 180도 회전 (마주보기)
q_R[0] = 0    # X축 오른쪽
q_R[1] = -0.2
q_R[2] = 0.0
q_R[3:7] = R.from_euler('z', 180, degrees=True).as_quat()

# =========================================================
# End-effector & Orientation
# =========================================================
EE_NAME = "link6"
ee_id_L = model_L.getFrameId(EE_NAME)
ee_id_R = model_R.getFrameId(EE_NAME)

# 왼쪽 팔의 기본 자세 (아래를 향함)
R_des_L = ee_orientation_from_euler(0, -180, 0)

R_des_R = ee_orientation_from_euler(0, 0, 0)

# 오른쪽 팔의 목표 자세 (완전한 거울 대칭을 위해 회전 변환 적용)
# 베이스가 180도 돌았으므로 목표 오리엔테이션도 그에 대응해야 함
R_des_R = R.from_euler('z', 0, degrees=True).as_matrix() @ R_des_R @ np.diag([1, -1, -1])

# =========================================================
# Initial Target
# =========================================================
pin.forwardKinematics(model_L, data_L, q_L)
pin.updateFramePlacements(model_L, data_L)
p_des_L = data_L.oMf[ee_id_L].translation.copy()

# 오른쪽 목표 위치는 왼쪽의 X축 대칭 (중앙 기준)
p_des_R = p_des_L.copy()
p_des_R[0] *= -1
p_des_R[1] *= -1


viz.display(q_L)
viz_R.display(q_R)

print("작동 시작: WASD (XY이동), QE (Z이동)")

# =========================================================
# Main loop
# =========================================================
dt = 0.02
step = 0.01

while True:
    # 키보드 입력에 따른 왼쪽 타겟 이동
    if keyboard.is_pressed("w"): p_des_L[0] += step
    if keyboard.is_pressed("s"): p_des_L[0] -= step
    if keyboard.is_pressed("a"): p_des_L[1] += step
    if keyboard.is_pressed("d"): p_des_L[1] -= step
    if keyboard.is_pressed("q"): p_des_L[2] += step
    if keyboard.is_pressed("e"): p_des_L[2] -= step

    # 오른쪽 타겟은 왼쪽의 X축 대칭 (거울)
    p_des_R = p_des_L.copy()
    p_des_R[0] *= -1
    p_des_R[1] *= -1

    # IK 계산
    q_L = step_ik(model_L, data_L, q_L, ee_id_L, p_des_L, R_des_L)
    q_R = step_ik(model_R, data_R, q_R, ee_id_R, p_des_R, R_des_R)

    # 시각화 업데이트
    viz.display(q_L)
    viz_R.display(q_R)

    time.sleep(dt)