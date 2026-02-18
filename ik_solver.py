# ik_solver.py
import numpy as np


def solve_ik_step(J: np.ndarray, pos_err: np.ndarray, rot_err: np.ndarray,
                  damping_base: float, dq_max: float,
                  q_cur: np.ndarray = None, q_home: np.ndarray = None,
                  null_gain: float = 0.0) -> np.ndarray:
    """Damped Least Squares IK step with optional null-space home pull.

    Args:
        J: (6, n_joints) Jacobian matrix
        pos_err: (3,) position error (target - current)
        rot_err: (3,) orientation error via log3
        damping_base: base damping factor for DLS
        dq_max: max joint velocity clamp
        q_cur: current joint angles (for null-space)
        q_home: home joint angles (for null-space)
        null_gain: null-space gain (0 to disable)

    Returns:
        dq: (n_joints,) joint velocity command, clamped to [-dq_max, dq_max]
    """
    err = np.hstack([pos_err, rot_err])
    damping = damping_base * (1.0 + np.linalg.norm(err))

    JT = J.T
    A = J @ JT + damping * np.eye(6)

    dq = JT @ np.linalg.solve(A, err)

    if q_cur is not None and q_home is not None and null_gain > 0:
        N = np.eye(J.shape[1]) - JT @ np.linalg.solve(A, J)
        dq += N @ (null_gain * (q_home - q_cur))

    return np.clip(dq, -dq_max, dq_max)
