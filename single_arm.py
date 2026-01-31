# -*- coding: utf-8 -*-
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R, Slerp
import keyboard

# =============================
# URDF & Mesh Paths
# =============================
URDF_PATH = r"C:\Users\kimha\PycharmProjects\study\test_arm\test_arm.urdf"
MESH_ROOT = r"C:\Users\kimha\PycharmProjects\study\test_arm"

# =============================
# Model + Data
# =============================
model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()

visual_model = pin.buildGeomFromUrdf(model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT])
viz = MeshcatVisualizer(model, None, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# =============================
# Utility functions
# =============================
def ee_orientation_from_euler(roll, pitch, yaw, degrees=True):
    """EE rotation matrix from Euler angles"""
    return R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees).as_matrix()

def slerp_rotation(R1, R2, alpha):
    """Slerp between two rotation matrices"""
    r1 = R.from_matrix(R1)
    r2 = R.from_matrix(R2)
    slerp = Slerp([0, 1], R.concatenate([r1, r2]))
    return slerp([alpha])[0].as_matrix()

# =============================
# Initial Joint State
# =============================
q = pin.neutral(model)
q[2] = 0.5
q[3:7] = R.from_euler('xyz', [0, 0, 0]).as_quat()

EE_NAME = "link6"
ee_id = model.getFrameId(EE_NAME)

# =============================
# Goal Orientation
# =============================
R_des = ee_orientation_from_euler(0, 180, 0)

# =============================
# Parameters
# =============================
dt = 0.02
step = 0.01
dq_max = 0.02  # max joint change per loop
workspace_limit = np.array([0.25, 0.25, 0.5])
joint_damping_base = 1e-3
null_space_gain = 0.1
q_home = q.copy()  # null-space reference

# =============================
# Initial target position
# =============================
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
p_des = data.oMf[ee_id].translation.copy()

# For trajectory visualization
ee_history = [p_des.copy()]

print("WASD: XY | Q/E: Z | link6 X축 = 아래 ↓")

# =============================
# Main loop
# =============================
while True:
    # ---- Forward kinematics ----
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[ee_id]
    p_cur = oMf.translation
    R_cur = oMf.rotation

    # ---- User input for position ----
    if keyboard.is_pressed("w"): p_des[0] += step
    if keyboard.is_pressed("s"): p_des[0] -= step
    if keyboard.is_pressed("a"): p_des[1] += step
    if keyboard.is_pressed("d"): p_des[1] -= step
    if keyboard.is_pressed("q"): p_des[2] += step
    if keyboard.is_pressed("e"): p_des[2] -= step

    # ---- Soft workspace limit ----
    alpha_ws = 0.5  # soft approach factor (0=hard, 1=soft)
    p_des = np.clip(p_des, -workspace_limit, workspace_limit)
    # optional: can add smooth ramp for soft approach

    # ---- Rotation error (log map) ----
    rot_err = pin.log3(R_des @ R_cur.T)
    pos_err = p_des - p_cur
    err = np.hstack([pos_err, rot_err])

    # ---- Adaptive damping ----
    damping = joint_damping_base * (1 + np.linalg.norm(err))

    # ---- Jacobian ----
    J6 = pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    J = J6[:, 6:]  # remove free-flyer

    # ---- Damped pseudo-inverse IK with null-space ----
    JT = J.T
    JJt = J @ JT
    dq = JT @ np.linalg.inv(JJt + damping * np.eye(6)) @ err

    # Null-space posture control
    dq += (np.eye(len(dq)) - JT @ np.linalg.inv(JJt + damping * np.eye(6)) @ J) @ (null_space_gain * (q_home[7:] - q[7:]))

    # ---- Velocity limit ----
    dq = np.clip(dq, -dq_max, dq_max)

    # ---- Update joint positions ----
    q[7:] += dq

    # ---- Soft joint limits ----
    q[7:] = np.clip(q[7:], model.lowerPositionLimit[7:], model.upperPositionLimit[7:])

    # ---- Visualize ----
    viz.display(q)

    # ---- Log trajectory ----
    ee_history.append(data.oMf[ee_id].translation.copy())

    time.sleep(dt)
