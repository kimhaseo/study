# -*- coding: utf-8 -*-
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R
import keyboard

from config import URDF_PATH, MESH_ROOT
from math_utils import rpy_deg_to_R
from ik_solver import solve_ik_step


def _step_ik(model, data, q, ee_id, p_des, R_des, damping=1e-3, dq_max=0.05):
    """Single IK step for standalone dual-arm loop."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    oMf = data.oMf[ee_id]
    p_cur = oMf.translation
    R_cur = oMf.rotation

    pos_err = p_des - p_cur
    rot_err = pin.log3(R_des @ R_cur.T)

    J6 = pin.computeFrameJacobian(
        model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    J = J6[:, 6:]

    dq = solve_ik_step(J, pos_err, rot_err, damping, dq_max)
    q[7:] += dq


def main():
    # Models (LEFT / RIGHT)
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

    # Initial q (mirror layout)
    q_L = pin.neutral(model_L)
    q_R = pin.neutral(model_R)

    q_L[0], q_L[1], q_L[2] = 0, 0.2, 0.0
    q_L[3:7] = R.identity().as_quat()

    q_R[0], q_R[1], q_R[2] = 0, -0.2, 0.0
    q_R[3:7] = R.from_euler("z", 180, degrees=True).as_quat()

    # End-effector
    EE_NAME = "link6"
    ee_id_L = model_L.getFrameId(EE_NAME)
    ee_id_R = model_R.getFrameId(EE_NAME)

    # Goal orientations
    R_des_L = rpy_deg_to_R(0, -180, 0)
    R_des_R = rpy_deg_to_R(0, 0, 0)
    R_des_R = R.from_euler("z", 0, degrees=True).as_matrix() @ R_des_R @ np.diag([1, -1, -1])

    # Initial target
    pin.forwardKinematics(model_L, data_L, q_L)
    pin.updateFramePlacements(model_L, data_L)
    p_des_L = data_L.oMf[ee_id_L].translation.copy()

    p_des_R = p_des_L.copy()
    p_des_R[0] *= -1
    p_des_R[1] *= -1

    viz.display(q_L)
    viz_R.display(q_R)

    print("WASD (XY), QE (Z)")

    # Main loop
    dt = 0.02
    step = 0.01

    while True:
        if keyboard.is_pressed("w"): p_des_L[0] += step
        if keyboard.is_pressed("s"): p_des_L[0] -= step
        if keyboard.is_pressed("a"): p_des_L[1] += step
        if keyboard.is_pressed("d"): p_des_L[1] -= step
        if keyboard.is_pressed("q"): p_des_L[2] += step
        if keyboard.is_pressed("e"): p_des_L[2] -= step

        # Mirror
        p_des_R = p_des_L.copy()
        p_des_R[0] *= -1
        p_des_R[1] *= -1

        # IK
        _step_ik(model_L, data_L, q_L, ee_id_L, p_des_L, R_des_L)
        _step_ik(model_R, data_R, q_R, ee_id_R, p_des_R, R_des_R)

        viz.display(q_L)
        viz_R.display(q_R)

        time.sleep(dt)


if __name__ == "__main__":
    main()
