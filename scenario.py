# scenario.py
import time
import numpy as np
from units import MM_TO_M, mm_to_m_vec
from math_utils import clamp_vec3, rot_error_log3, rpy_deg_to_R

class ScenarioPlayer:
    def __init__(self, workspace_limit_m: np.ndarray):
        self.workspace_limit_m = workspace_limit_m

    def apply_waypoint_to_goal(self, shared, idx: int):
        if idx < 0 or idx >= len(shared.waypoints):
            return
        wp = shared.waypoints[idx]
        p_m = clamp_vec3(mm_to_m_vec(wp["p_mm"].copy()), self.workspace_limit_m)
        rpy = wp["rpy"].copy()
        R_goal = rpy_deg_to_R(rpy[0], rpy[1], rpy[2])

        shared.p_goal[:] = p_m
        shared.rpy_goal_deg[:] = rpy
        shared.R_goal[:, :] = R_goal
        shared.last_reach_time = 0.0

    def step(self, shared, p_cur: np.ndarray, R_cur: np.ndarray):
        if not shared.play_enable or len(shared.waypoints) == 0:
            return

        # Decide reach based on EE vs GOAL
        p_goal = shared.p_goal.copy()
        R_goal = shared.R_goal.copy()

        pos_err = p_goal - p_cur
        rot_err = rot_error_log3(R_goal, R_cur)

        reached = False
        if shared.play_mode == "hold":
            if shared.last_reach_time == 0.0:
                shared.last_reach_time = time.time()
            if (time.time() - shared.last_reach_time) >= shared.hold_time_s:
                reached = True
        else:
            ptol_m = shared.pos_tol_mm * MM_TO_M
            if np.linalg.norm(pos_err) <= ptol_m and np.linalg.norm(rot_err) <= shared.rot_tol_rad:
                if shared.last_reach_time == 0.0:
                    shared.last_reach_time = time.time()
                if (time.time() - shared.last_reach_time) >= 0.15:
                    reached = True
            else:
                shared.last_reach_time = 0.0

        if reached:
            nxt = shared.play_idx + 1
            if nxt >= len(shared.waypoints):
                shared.play_enable = False
            else:
                shared.play_idx = nxt
                shared.last_reach_time = 0.0
                self.apply_waypoint_to_goal(shared, nxt)
