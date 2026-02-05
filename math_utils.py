# math_utils.py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R, Slerp

def rpy_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()

def R_to_rpy_deg(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(Rm).as_euler("xyz", degrees=True)

def clamp_vec3(v: np.ndarray, lim: np.ndarray) -> np.ndarray:
    return np.clip(v, -lim, lim)

def rot_error_log3(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    return pin.log3(R_des @ R_cur.T)

def slerp_R(R1: np.ndarray, R2: np.ndarray, alpha_0_1: float) -> np.ndarray:
    if alpha_0_1 <= 0.0:
        return R1
    if alpha_0_1 >= 1.0:
        return R2
    r1 = R.from_matrix(R1)
    r2 = R.from_matrix(R2)
    s = Slerp([0.0, 1.0], R.concatenate([r1, r2]))
    return s([alpha_0_1])[0].as_matrix()
