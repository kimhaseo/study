# controller.py
import time
import threading
import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
    _pynput_available = True
except ImportError:
    _pynput_available = False

from units import MM_TO_M
from math_utils import clamp_vec3, rot_error_log3, slerp_R
from ik_solver import solve_ik_step

class IKController:
    def __init__(self, robot, shared, workspace_limit_m: np.ndarray, scenario):
        self.robot = robot
        self.shared = shared
        self.workspace_limit_m = workspace_limit_m
        self.scenario = scenario

        self.stop_flag = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

        # pynput 키보드 리스너
        self._pressed_keys: set = set()
        self._kb_listener = None
        if _pynput_available:
            try:
                self._kb_listener = pynput_keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                self._kb_listener.start()
            except Exception:
                self._kb_listener = None

    def _on_key_press(self, key):
        try:
            self._pressed_keys.add(key.char)
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            self._pressed_keys.discard(key.char)
        except AttributeError:
            pass

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        if self._kb_listener is not None:
            self._kb_listener.stop()

    def _loop(self):
        last_print = 0.0

        while not self.stop_flag:
            # FK + current EE
            p_cur, R_cur = self.robot.get_ee_pose()

            # lock 1: EE 게시 + 파라미터 스냅샷만
            with self.shared.lock:
                self.shared.latest_p[:] = p_cur
                self.shared.latest_R[:, :] = R_cur

                kb           = bool(self.shared.keyboard_enable)
                dt           = float(self.shared.dt)
                local_step_m = float(self.shared.step_mm) * MM_TO_M
                dq_max       = float(self.shared.dq_max)
                damp_base    = float(self.shared.damping_base)
                ns_gain      = float(self.shared.null_gain)
                vmax_m_s     = float(self.shared.v_cmd_max_mm_s) * MM_TO_M
                wmax_deg_s   = float(self.shared.w_cmd_max_deg_s)
                pe_now       = bool(self.shared.play_enable)
                p_cmd        = self.shared.p_cmd.copy()
                R_cmd        = self.shared.R_cmd.copy()

            # lock 2: scenario.step() + 목표값 스냅샷 (step이 shared 직접 수정하므로 lock 필요)
            with self.shared.lock:
                self.scenario.step(self.shared, p_cur, R_cur)
                p_goal = self.shared.p_goal.copy()
                R_goal = self.shared.R_goal.copy()

            # 키보드로 GOAL 업데이트 (시나리오 재생 중에는 무시)
            if kb and (not pe_now) and self._kb_listener is not None:
                if 'w' in self._pressed_keys: p_goal[0] += local_step_m
                if 's' in self._pressed_keys: p_goal[0] -= local_step_m
                if 'a' in self._pressed_keys: p_goal[1] += local_step_m
                if 'd' in self._pressed_keys: p_goal[1] -= local_step_m
                if 'q' in self._pressed_keys: p_goal[2] += local_step_m
                if 'e' in self._pressed_keys: p_goal[2] -= local_step_m

                p_goal = clamp_vec3(p_goal, self.workspace_limit_m)
                with self.shared.lock:
                    self.shared.p_goal[:] = p_goal

            # ===== Linear CMD rate-limit toward GOAL =====
            dp = p_goal - p_cmd
            dist = float(np.linalg.norm(dp))
            max_step = vmax_m_s * dt
            if dist <= max_step or dist < 1e-12:
                p_cmd = p_goal.copy()
            else:
                p_cmd = p_cmd + (dp / dist) * max_step
            p_cmd = clamp_vec3(p_cmd, self.workspace_limit_m)

            # orientation rate-limit via SLERP
            angle = float(np.linalg.norm(rot_error_log3(R_goal, R_cmd)))
            wmax = np.deg2rad(wmax_deg_s)
            max_ang = wmax * dt
            if angle <= max_ang or angle < 1e-9:
                R_cmd = R_goal.copy()
            else:
                alpha = max_ang / angle
                R_cmd = slerp_R(R_cmd, R_goal, alpha)

            with self.shared.lock:
                self.shared.p_cmd[:] = p_cmd
                self.shared.R_cmd[:, :] = R_cmd

            # ===== IK DLS =====
            pos_err = p_cmd - p_cur
            rot_err = rot_error_log3(R_cmd, R_cur)
            J = self.robot.compute_jacobian_joints()

            dq = solve_ik_step(
                J, pos_err, rot_err, damp_base, dq_max,
                q_cur=self.robot.q[7:], q_home=self.robot.q_home[7:],
                null_gain=ns_gain,
            )

            self.robot.apply_dq(dq, dq_max)
            self.robot.display()

            now = time.time()
            if now - last_print > 0.2:
                last_print = now
                print(f"EE pos [m] X:{p_cur[0]: .4f} Y:{p_cur[1]: .4f} Z:{p_cur[2]: .4f}")

            time.sleep(dt)
