# math_utils.py
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


def rpy_deg_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    return R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()


def R_to_rpy_deg(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(Rm).as_euler("xyz", degrees=True)


def clamp_vec3(v: np.ndarray, lim: np.ndarray) -> np.ndarray:
    return np.clip(v, -lim, lim)


def rot_error_log3(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    return pin.log3(R_des @ R_cur.T)


def slerp_R(R1: np.ndarray, R2: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return R1.copy()
    if alpha >= 1.0:
        return R2.copy()
    # Direct slerp via pinocchio log3/exp3 — no scipy Slerp object overhead
    log_rel = pin.log3(R1.T @ R2)
    return R1 @ pin.exp3(alpha * log_rel)
