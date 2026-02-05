# shared_state.py
import threading
import numpy as np
from dataclasses import dataclass, field

@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Targets (internal meters)
    p_goal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R_goal: np.ndarray = field(default_factory=lambda: np.eye(3))

    p_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R_cmd: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Goal RPY stored (deg) for UI display
    rpy_goal_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Latest EE pose
    latest_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    latest_R: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Control params
    keyboard_enable: bool = True
    dt: float = 0.01
    step_mm: float = 5.0
    dq_max: float = 0.05
    damping_base: float = 1e-3
    null_gain: float = 0.1

    v_cmd_max_mm_s: float = 500.0
    w_cmd_max_deg_s: float = 90.0

    # Scenario
    waypoints: list = field(default_factory=list)  # each: dict(name, p_mm(3), rpy_deg(3))
    play_enable: bool = False
    play_idx: int = 0
    play_mode: str = "hold"  # hold/error
    hold_time_s: float = 1.0
    pos_tol_mm: float = 3.0
    rot_tol_rad: float = 0.03
    last_reach_time: float = 0.0
