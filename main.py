# main.py
from config import (
    URDF_PATH, MESH_ROOT, EE_NAME,
    WORKSPACE_LIMIT_M, DEFAULT_RPY_DEG,
    DT, STEP_MM, DQ_MAX, JOINT_DAMPING_BASE, NULL_SPACE_GAIN,
    V_CMD_MAX_MM_S, W_CMD_MAX_DEG_S,
    HOLD_TIME_S, POS_TOL_MM, ROT_TOL_RAD
)

from math_utils import rpy_deg_to_R, clamp_vec3
from robot_model import RobotModel
from shared_state import SharedState
from ui import IKUI
from controller import IKController
try:
    from camera import D405Detector
    _camera_available = True
except ImportError:
    _camera_available = False

def main():
    robot = RobotModel(URDF_PATH, MESH_ROOT, EE_NAME)
    detector = D405Detector("yolov8n.pt") if _camera_available else None

    shared = SharedState()
    shared.dt = DT
    shared.step_mm = STEP_MM
    shared.dq_max = DQ_MAX
    shared.damping_base = JOINT_DAMPING_BASE
    shared.null_gain = NULL_SPACE_GAIN
    shared.v_cmd_max_mm_s = V_CMD_MAX_MM_S
    shared.w_cmd_max_deg_s = W_CMD_MAX_DEG_S

    shared.hold_time_s = HOLD_TIME_S
    shared.pos_tol_mm = POS_TOL_MM
    shared.rot_tol_rad = ROT_TOL_RAD

    # Init shared targets from current EE
    p0, R0 = robot.get_ee_pose()
    p0 = clamp_vec3(p0, WORKSPACE_LIMIT_M)
    shared.p_goal[:] = p0
    shared.p_cmd[:] = p0

    rpy0 = DEFAULT_RPY_DEG
    shared.rpy_goal_deg[:] = rpy0
    shared.R_goal[:, :] = rpy_deg_to_R(*rpy0)
    shared.R_cmd[:, :] = shared.R_goal.copy()

    # controller thread
    ctrl = IKController(robot, shared, WORKSPACE_LIMIT_M)
    ctrl.start()

    # UI
    ui = IKUI(shared, WORKSPACE_LIMIT_M, detector)
    try:
        ui.mainloop()
    finally:
        ctrl.stop()
        if detector is not None:
            detector.close()

if __name__ == "__main__":
    main()
