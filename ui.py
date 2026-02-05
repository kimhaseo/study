# ui.py
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from units import m_to_mm_vec, mm_to_m_vec, M_TO_MM
from math_utils import rpy_deg_to_R, R_to_rpy_deg, clamp_vec3
from scenario import ScenarioPlayer

class IKUI:
    def __init__(self, shared, workspace_limit_m: np.ndarray):
        self.shared = shared
        self.workspace_limit_m = workspace_limit_m
        self.workspace_limit_mm = workspace_limit_m * M_TO_MM
        self.scenario = ScenarioPlayer(workspace_limit_m)

        self.root = tk.Tk()
        self.root.title("Pinocchio IK UI (mm + deg) + Scenario (modular)")

        # UI vars
        with self.shared.lock:
            p_goal_mm = m_to_mm_vec(self.shared.p_goal.copy())
            rpy = self.shared.rpy_goal_deg.copy()

        self.var_x = tk.DoubleVar(value=float(p_goal_mm[0]))
        self.var_y = tk.DoubleVar(value=float(p_goal_mm[1]))
        self.var_z = tk.DoubleVar(value=float(p_goal_mm[2]))

        self.var_roll = tk.DoubleVar(value=float(rpy[0]))
        self.var_pitch = tk.DoubleVar(value=float(rpy[1]))
        self.var_yaw = tk.DoubleVar(value=float(rpy[2]))

        self.var_step_mm = tk.DoubleVar(value=4.0)
        self.var_dqmax = tk.DoubleVar(value=0.05)
        self.var_damp = tk.DoubleVar(value=1e-4)
        self.var_ns = tk.DoubleVar(value=0.1)
        self.var_kb = tk.BooleanVar(value=True)

        self.var_vmax_mm_s = tk.DoubleVar(value=300.0)
        self.var_wmax_deg_s = tk.DoubleVar(value=90.0)

        self.var_hold = tk.DoubleVar(value=1.0)
        self.var_postol_mm = tk.DoubleVar(value=3.0)
        self.var_rottol = tk.DoubleVar(value=0.03)
        self.var_playmode = tk.StringVar(value="hold")
        self.var_record_source = tk.StringVar(value="target")

        self.lbl_cur = ttk.Label(self.root, text="EE cur [mm]: ...")
        self.lbl_goal = ttk.Label(self.root, text="GOAL [mm]: ...")
        self.lbl_cmd = ttk.Label(self.root, text="CMD  [mm]: ...")
        self.lbl_err = ttk.Label(self.root, text="ERR  : ...")
        self.lbl_play = ttk.Label(self.root, text="Scenario: stopped")

        self.list_wp = tk.Listbox(self.root, height=10, exportselection=False)

        self._build_layout()
        self.root.after(100, self._tick)

    def mainloop(self):
        self.root.mainloop()

    # -----------------------------
    # UI Actions
    # -----------------------------
    def apply_goal_from_ui(self):
        with self.shared.lock:
            p_mm = np.array([self.var_x.get(), self.var_y.get(), self.var_z.get()], dtype=float)
            p_m = clamp_vec3(mm_to_m_vec(p_mm), self.workspace_limit_m)
            self.shared.p_goal[:] = p_m

            rpy = np.array([self.var_roll.get(), self.var_pitch.get(), self.var_yaw.get()], dtype=float)
            self.shared.rpy_goal_deg[:] = rpy
            self.shared.R_goal[:, :] = rpy_deg_to_R(rpy[0], rpy[1], rpy[2])

            self.shared.step_mm = float(self.var_step_mm.get())
            self.shared.dq_max = float(self.var_dqmax.get())
            self.shared.damping_base = float(self.var_damp.get())
            self.shared.null_gain = float(self.var_ns.get())
            self.shared.keyboard_enable = bool(self.var_kb.get())

            self.shared.v_cmd_max_mm_s = float(self.var_vmax_mm_s.get())
            self.shared.w_cmd_max_deg_s = float(self.var_wmax_deg_s.get())

    def sync_goal_cmd_to_current(self):
        with self.shared.lock:
            p = clamp_vec3(self.shared.latest_p.copy(), self.workspace_limit_m)
            self.shared.p_goal[:] = p
            self.shared.p_cmd[:] = p
            self.shared.last_reach_time = 0.0
        self._sync_entries_from_goal()

    def home_nullspace(self):
        # EE는 유지(현재), null-space로 관절 홈 당김
        self.sync_goal_cmd_to_current()

    def quit(self):
        self.root.quit()
        self.root.destroy()

    # -----------------------------
    # Waypoints / Scenario
    # -----------------------------
    def refresh_waypoints(self):
        self.list_wp.delete(0, tk.END)
        with self.shared.lock:
            wps = list(self.shared.waypoints)
        for i, wp in enumerate(wps):
            pmm = wp["p_mm"]; rpy = wp["rpy"]
            self.list_wp.insert(
                tk.END,
                f"{i:02d} | {wp['name']} | p=({pmm[0]:.1f},{pmm[1]:.1f},{pmm[2]:.1f})mm rpy=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})"
            )

    def record_waypoint(self):
        with self.shared.lock:
            src = self.var_record_source.get()
            if src == "ee":
                p_m = self.shared.latest_p.copy()
                rpy = R_to_rpy_deg(self.shared.latest_R.copy())
            else:
                p_m = self.shared.p_goal.copy()
                rpy = self.shared.rpy_goal_deg.copy()

            p_m = clamp_vec3(p_m, self.workspace_limit_m)
            p_mm = m_to_mm_vec(p_m)

            self.shared.waypoints.append({
                "name": f"WP{len(self.shared.waypoints)}",
                "p_mm": p_mm.astype(float),
                "rpy": np.array(rpy, dtype=float),
            })
        self.refresh_waypoints()

    def goto_selected(self):
        sel = self.list_wp.curselection()
        if not sel:
            return
        idx = sel[0]
        with self.shared.lock:
            self.shared.play_enable = False
            self.shared.play_idx = idx
            self.scenario.apply_waypoint_to_goal(self.shared, idx)
        self._sync_entries_from_goal()

    def delete_selected(self):
        sel = self.list_wp.curselection()
        if not sel:
            return
        idx = sel[0]
        with self.shared.lock:
            if 0 <= idx < len(self.shared.waypoints):
                self.shared.waypoints.pop(idx)
        self.refresh_waypoints()

    def clear_all(self):
        with self.shared.lock:
            self.shared.waypoints.clear()
        self.refresh_waypoints()

    def play(self):
        with self.shared.lock:
            if len(self.shared.waypoints) == 0:
                messagebox.showinfo("Scenario", "웨이포인트가 없습니다. Record WP로 먼저 저장하세요.")
                return

            self.shared.play_mode = self.var_playmode.get()
            self.shared.hold_time_s = float(self.var_hold.get())
            self.shared.pos_tol_mm = float(self.var_postol_mm.get())
            self.shared.rot_tol_rad = float(self.var_rottol.get())

            self.shared.play_enable = True
            self.shared.play_idx = 0
            self.shared.last_reach_time = 0.0
            self.scenario.apply_waypoint_to_goal(self.shared, 0)

        self._sync_entries_from_goal()

    def stop(self):
        with self.shared.lock:
            self.shared.play_enable = False

    def save_csv(self):
        with self.shared.lock:
            if len(self.shared.waypoints) == 0:
                return
            wps = list(self.shared.waypoints)

        path = filedialog.asksaveasfilename(
            title="Save waypoints (mm)",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write("name,px_mm,py_mm,pz_mm,roll_deg,pitch_deg,yaw_deg\n")
            for wp in wps:
                p = wp["p_mm"]; rpy = wp["rpy"]
                f.write(f"{wp['name']},{p[0]},{p[1]},{p[2]},{rpy[0]},{rpy[1]},{rpy[2]}\n")

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Load waypoints (mm)",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        loaded = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            if not lines:
                return
            header = lines[0].replace(" ", "").lower()
            start = 1 if header.startswith("name,px") else 0

            for ln in lines[start:]:
                if not ln.strip():
                    continue
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) < 7:
                    continue
                name = parts[0]
                pmm = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
                rpy = np.array([float(parts[4]), float(parts[5]), float(parts[6])], dtype=float)

                # clamp
                p_m = clamp_vec3(mm_to_m_vec(pmm), self.workspace_limit_m)
                pmm = m_to_mm_vec(p_m)
                loaded.append({"name": name, "p_mm": pmm, "rpy": rpy})

        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        with self.shared.lock:
            self.shared.waypoints[:] = loaded
        self.refresh_waypoints()

    # -----------------------------
    # Layout & Tick
    # -----------------------------
    def _build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        # LEFT
        frm = ttk.LabelFrame(main, text="Target Control (mm / deg)", padding=10)
        frm.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Goal Position [mm]").grid(row=0, column=0, columnspan=3, sticky="w")

        self._row_entry(frm, 1, "X", self.var_x, f"limit ±{self.workspace_limit_mm[0]:.0f}")
        self._row_entry(frm, 2, "Y", self.var_y, f"limit ±{self.workspace_limit_mm[1]:.0f}")
        self._row_entry(frm, 3, "Z", self.var_z, f"limit ±{self.workspace_limit_mm[2]:.0f}")

        ttk.Label(frm, text="Goal Orientation RPY [deg]").grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self._row_entry(frm, 5, "Roll", self.var_roll, "")
        self._row_entry(frm, 6, "Pitch", self.var_pitch, "")
        self._row_entry(frm, 7, "Yaw", self.var_yaw, "")

        ttk.Label(frm, text="Control Params").grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 0))
        self._row_entry(frm, 9, "step [mm/key]", self.var_step_mm, "")
        self._row_entry(frm, 10, "dq_max", self.var_dqmax, "")
        self._row_entry(frm, 11, "damping_base", self.var_damp, "")
        self._row_entry(frm, 12, "null_gain", self.var_ns, "")
        ttk.Checkbutton(frm, text="Keyboard Enable (WASD/QE)", variable=self.var_kb).grid(row=13, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._row_entry(frm, 14, "vmax [mm/s]", self.var_vmax_mm_s, "")
        self._row_entry(frm, 15, "wmax [deg/s]", self.var_wmax_deg_s, "")

        btns = ttk.Frame(frm)
        btns.grid(row=16, column=0, columnspan=3, sticky="we", pady=(10, 0))
        for i in range(3):
            btns.columnconfigure(i, weight=1)

        ttk.Button(btns, text="Apply Goal", command=self.apply_goal_from_ui).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Button(btns, text="Sync goal/cmd = current EE", command=self.sync_goal_cmd_to_current).grid(row=0, column=1, sticky="we", padx=(0, 6))
        ttk.Button(btns, text="Home (null-space)", command=self.home_nullspace).grid(row=0, column=2, sticky="we")

        ttk.Button(frm, text="Quit", command=self.quit).grid(row=17, column=0, columnspan=3, sticky="we", pady=(8, 0))

        self.lbl_cur.grid(row=18, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.lbl_goal.grid(row=19, column=0, columnspan=3, sticky="w")
        self.lbl_cmd.grid(row=20, column=0, columnspan=3, sticky="w")
        self.lbl_err.grid(row=21, column=0, columnspan=3, sticky="w")
        self.lbl_play.grid(row=22, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # RIGHT
        sc = ttk.LabelFrame(main, text="Scenario (mm / deg)", padding=10)
        sc.grid(row=0, column=1, sticky="nsew")
        sc.columnconfigure(0, weight=1)
        sc.rowconfigure(1, weight=1)

        toprow = ttk.Frame(sc)
        toprow.grid(row=0, column=0, sticky="we")
        ttk.Label(toprow, text="Record source:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(toprow, text="Goal(Target)", value="target", variable=self.var_record_source).grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Radiobutton(toprow, text="EE(Current)", value="ee", variable=self.var_record_source).grid(row=0, column=2, sticky="w", padx=(8, 0))

        self.list_wp.grid(row=1, column=0, sticky="nsew", pady=(8, 8))

        wp_btns = ttk.Frame(sc)
        wp_btns.grid(row=2, column=0, sticky="we")
        for i in range(6):
            wp_btns.columnconfigure(i, weight=1)

        ttk.Button(wp_btns, text="Record WP", command=self.record_waypoint).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Button(wp_btns, text="Goto selected", command=self.goto_selected).grid(row=0, column=1, sticky="we", padx=(0, 6))
        ttk.Button(wp_btns, text="Delete", command=self.delete_selected).grid(row=0, column=2, sticky="we", padx=(0, 6))
        ttk.Button(wp_btns, text="Clear", command=self.clear_all).grid(row=0, column=3, sticky="we", padx=(0, 6))
        ttk.Button(wp_btns, text="Save CSV", command=self.save_csv).grid(row=0, column=4, sticky="we", padx=(0, 6))
        ttk.Button(wp_btns, text="Load CSV", command=self.load_csv).grid(row=0, column=5, sticky="we")

        playfrm = ttk.Frame(sc)
        playfrm.grid(row=3, column=0, sticky="we", pady=(8, 0))
        for i in range(6):
            playfrm.columnconfigure(i, weight=1)

        ttk.Label(playfrm, text="Play mode:").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(playfrm, text="Hold(s)", value="hold", variable=self.var_playmode).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(playfrm, text="Error tol", value="error", variable=self.var_playmode).grid(row=0, column=2, sticky="w")

        ttk.Label(playfrm, text="Hold [s]").grid(row=1, column=0, sticky="w")
        ttk.Entry(playfrm, textvariable=self.var_hold, width=8).grid(row=1, column=1, sticky="w")

        ttk.Label(playfrm, text="pos_tol [mm]").grid(row=1, column=2, sticky="w")
        ttk.Entry(playfrm, textvariable=self.var_postol_mm, width=8).grid(row=1, column=3, sticky="w")

        ttk.Label(playfrm, text="rot_tol [rad]").grid(row=1, column=4, sticky="w")
        ttk.Entry(playfrm, textvariable=self.var_rottol, width=8).grid(row=1, column=5, sticky="w")

        playbtns = ttk.Frame(sc)
        playbtns.grid(row=4, column=0, sticky="we", pady=(8, 0))
        playbtns.columnconfigure(0, weight=1)
        playbtns.columnconfigure(1, weight=1)
        ttk.Button(playbtns, text="Play", command=self.play).grid(row=0, column=0, sticky="we", padx=(0, 6))
        ttk.Button(playbtns, text="Stop", command=self.stop).grid(row=0, column=1, sticky="we")

        # initial list
        self.refresh_waypoints()

    def _row_entry(self, parent, row, label, var, hint):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="w")
        if hint:
            ttk.Label(parent, text=hint).grid(row=row, column=2, sticky="w")

    def _sync_entries_from_goal(self):
        with self.shared.lock:
            p_goal_mm = m_to_mm_vec(self.shared.p_goal.copy())
            rpy = self.shared.rpy_goal_deg.copy()
        self.var_x.set(float(p_goal_mm[0]))
        self.var_y.set(float(p_goal_mm[1]))
        self.var_z.set(float(p_goal_mm[2]))
        self.var_roll.set(float(rpy[0]))
        self.var_pitch.set(float(rpy[1]))
        self.var_yaw.set(float(rpy[2]))

    def _tick(self):
        with self.shared.lock:
            pc = self.shared.latest_p.copy()
            pd_goal = self.shared.p_goal.copy()
            pd_cmd = self.shared.p_cmd.copy()
            rpy_goal = self.shared.rpy_goal_deg.copy()
            rpy_cmd = R_to_rpy_deg(self.shared.R_cmd.copy())
            pe = self.shared.play_enable
            idx = self.shared.play_idx
            nwp = len(self.shared.waypoints)
            mode = self.shared.play_mode

        pc_mm = m_to_mm_vec(pc)
        goal_mm = m_to_mm_vec(pd_goal)
        cmd_mm = m_to_mm_vec(pd_cmd)
        err_mm = float(np.linalg.norm(cmd_mm - pc_mm))

        self.lbl_cur.config(text=f"EE cur [mm]: ({pc_mm[0]:.2f}, {pc_mm[1]:.2f}, {pc_mm[2]:.2f})")
        self.lbl_goal.config(text=f"GOAL [mm]: ({goal_mm[0]:.2f}, {goal_mm[1]:.2f}, {goal_mm[2]:.2f}) | rpy=({rpy_goal[0]:.1f},{rpy_goal[1]:.1f},{rpy_goal[2]:.1f})")
        self.lbl_cmd.config(text=f"CMD  [mm]: ({cmd_mm[0]:.2f}, {cmd_mm[1]:.2f}, {cmd_mm[2]:.2f}) | rpy=({rpy_cmd[0]:.1f},{rpy_cmd[1]:.1f},{rpy_cmd[2]:.1f})")
        self.lbl_err.config(text=f"ERR: |CMD-EE| = {err_mm:.3f} mm")

        if pe:
            self.lbl_play.config(text=f"Scenario: PLAYING idx={idx}/{max(nwp-1,0)} mode={mode}")
        else:
            self.lbl_play.config(text="Scenario: stopped")

        self.root.after(100, self._tick)
