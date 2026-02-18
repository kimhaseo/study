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


def main():
    # Model + Data
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()

    visual_model = pin.buildGeomFromUrdf(
        model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT]
    )
    viz = MeshcatVisualizer(model, None, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Initial joint state
    q = pin.neutral(model)
    q[2] = 0.5
    q[3:7] = R.from_euler("xyz", [0, 0, 0]).as_quat()
    q_home = q.copy()

    EE_NAME = "link6"
    ee_id = model.getFrameId(EE_NAME)

    # Goal orientation
    R_des = rpy_deg_to_R(0, 180, 0)

    # Parameters
    dt = 0.02
    step = 0.01
    dq_max = 0.02
    workspace_limit = np.array([0.25, 0.25, 0.5])
    damping_base = 1e-3
    null_gain = 0.1

    # Initial target position
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_des = data.oMf[ee_id].translation.copy()

    print("WASD: XY | Q/E: Z | link6 X축 = 아래")

    # Main loop
    while True:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf = data.oMf[ee_id]
        p_cur = oMf.translation
        R_cur = oMf.rotation

        # Keyboard input
        if keyboard.is_pressed("w"): p_des[0] += step
        if keyboard.is_pressed("s"): p_des[0] -= step
        if keyboard.is_pressed("a"): p_des[1] += step
        if keyboard.is_pressed("d"): p_des[1] -= step
        if keyboard.is_pressed("q"): p_des[2] += step
        if keyboard.is_pressed("e"): p_des[2] -= step

        p_des = np.clip(p_des, -workspace_limit, workspace_limit)

        # IK
        pos_err = p_des - p_cur
        rot_err = pin.log3(R_des @ R_cur.T)

        J6 = pin.computeFrameJacobian(
            model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J6[:, 6:]

        dq = solve_ik_step(
            J, pos_err, rot_err, damping_base, dq_max,
            q_cur=q[7:], q_home=q_home[7:], null_gain=null_gain,
        )

        q[7:] += dq
        q[7:] = np.clip(q[7:], model.lowerPositionLimit[7:], model.upperPositionLimit[7:])

        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    main()
