# -*- coding: utf-8 -*-
import time
import threading
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R, Slerp
import keyboard
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# =============================
# Units
# =============================
MM_TO_M = 1e-3
M_TO_MM = 1e3

# =============================
# URDF & Mesh Paths
# =============================
URDF_PATH = r"/test_arm/test_arm.urdf"
MESH_ROOT = r"C:\Users\kimha\PycharmProjects\study\test_arm"

# =============================
# Model + Data
# =============================
model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()

visual_model = pin.buildGeomFromUrdf(
    model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT]
)
viz = MeshcatVisualizer(model, None, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# =============================
# Utility
# =============================
def rpy_deg_to_R(roll_deg, pitch_deg, yaw_deg):
    return R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_matrix()

def R_to_rpy_deg(Rm):
    return R.from_matrix(Rm).as_euler("xyz", degrees=True)

def clamp_vec3(v, lim):
    return np.clip(v, -lim, lim)

def rot_error_log3(R_des, R_cur):
    return pin.log3(R_des @ R_cur.T)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def slerp_R(R1, R2, alpha_0_1):
    """Slerp between two rotation matrices"""
    if alpha_0_1 <= 0.0:
        return R1
    if alpha_0_1 >= 1.0:
        return R2
    r1 = R.from_matrix(R1)
    r2 = R.from_matrix(R2)
    s = Slerp([0.0, 1.0], R.concatenate([r1, r2]))
    return s([alpha_0_1])[0].as_matrix()

def m_to_mm_vec(v_m):
    return v_m * M_TO_MM

def mm_to_m_vec(v_mm):
    return v_mm * MM_TO_M

# =============================
# Initial Joint State
# =============================
q = pin.neutral(model)
q[2] = 0.5
q[3:7] = R.from_euler("xyz", [0, 0, 0]).as_quat()

EE_NAME = "link6"
ee_id = model.getFrameId(EE_NAME)

# FK init
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
oMf = data.oMf[ee_id]
p_cur0 = oMf.translation.copy()
R_cur0 = oMf.rotation.copy()

# =============================
# Shared state (UI <-> control loop)
# =============================
lock = threading.Lock()

# workspace limit in meters (internal)
workspace_limit_m = np.array([0.25, 0.25, 0.5], dtype=float)
workspace_limit_mm = workspace_limit_m * M_TO_MM

# ---- Goal (final target) set by UI/keyboard/scenario (internal: meters)
p_goal_shared = clamp_vec3(p_cur0.copy(), workspace_limit_m)
rpy_goal_shared = np.array([0.0, -90.0, 0.0], dtype=float)
R_goal_shared = rpy_deg_to_R(*rpy_goal_shared)

# ---- Command (intermediate target) followed by IK (internal: meters)
p_cmd_shared = p_goal_shared.copy()
R_cmd_shared = R_goal_shared.copy()

# UI flags / params
keyboard_enable = True
dt = 0.01

# step is now mm/key in UI; internal uses meters
step_mm = 5.0
dq_max = 0.05
joint_damping_base = 1e-3
null_space_gain = 0.1

# Linear / angular command rates
v_cmd_max_mm_s = 500.0   # mm/s  (직선 이동 속도)
w_cmd_max_deg = 90.0     # deg/s

q_home = q.copy()

# =============================
# Scenario / Waypoints (shared)
# =============================
# Each waypoint: dict(name, p_mm(3), rpy_deg(3))
waypoints = []
play_enable = False
play_idx = 0
play_mode = "hold"  # "hold" or "error"
hold_time = 1.0
pos_tol_mm = 3.0
rot_tol = 0.03
last_reach_time = 0.0

# latest EE
latest_p_cur = p_cur0.copy()
latest_R_cur = R_cur0.copy()

# =============================
# Tkinter UI
# =============================
root = tk.Tk()
root.title("Pinocchio IK Target UI (mm + deg) + Scenario + Linear EE")

# Variables (UI shows mm)
var_x = tk.DoubleVar(value=float(p_goal_shared[0] * M_TO_MM))
var_y = tk.DoubleVar(value=float(p_goal_shared[1] * M_TO_MM))
var_z = tk.DoubleVar(value=float(p_goal_shared[2] * M_TO_MM))

var_roll  = tk.DoubleVar(value=float(rpy_goal_shared[0]))
var_pitch = tk.DoubleVar(value=float(rpy_goal_shared[1]))
var_yaw   = tk.DoubleVar(value=float(rpy_goal_shared[2]))

# Params (UI)
var_step_mm = tk.DoubleVar(value=float(step_mm))
var_dqmax = tk.DoubleVar(value=float(dq_max))
var_damp = tk.DoubleVar(value=float(joint_damping_base))
var_ns = tk.DoubleVar(value=float(null_space_gain))
var_kb = tk.BooleanVar(value=True)

# Linear control params (UI)
var_vmax_mm_s = tk.DoubleVar(value=float(v_cmd_max_mm_s))
var_wmax = tk.DoubleVar(value=float(w_cmd_max_deg))

# Scenario vars (UI)
var_hold = tk.DoubleVar(value=float(hold_time))
var_postol_mm = tk.DoubleVar(value=float(pos_tol_mm))
var_rottol = tk.DoubleVar(value=float(rot_tol))
var_playmode = tk.StringVar(value="hold")
var_record_source = tk.StringVar(value="target")  # "target" or "ee"

lbl_cur = ttk.Label(root, text="EE cur: (x, y, z) [mm]")
lbl_goal = ttk.Label(root, text="GOAL: (x, y, z) [mm] | rpy(deg)")
lbl_cmd  = ttk.Label(root, text="CMD : (x, y, z) [mm] | rpy(deg)")
lbl_err  = ttk.Label(root, text="ERR : pos [mm]")
lbl_play = ttk.Label(root, text="Scenario: stopped")

list_wp = tk.Listbox(root, height=10, exportselection=False)

def sync_ui_entries_from_goal():
    with lock:
        p_m = p_goal_shared.copy()
        rpy = rpy_goal_shared.copy()
    p_mm = m_to_mm_vec(p_m)
    var_x.set(float(p_mm[0])); var_y.set(float(p_mm[1])); var_z.set(float(p_mm[2]))
    var_roll.set(float(rpy[0])); var_pitch.set(float(rpy[1])); var_yaw.set(float(rpy[2]))

def ui_apply_target():
    """UI(mm/deg) -> GOAL(m/deg). CMD will move linearly toward GOAL."""
    global dq_max, joint_damping_base, null_space_gain, keyboard_enable
    global step_mm, v_cmd_max_mm_s, w_cmd_max_deg

    with lock:
        # position mm -> m
        p_mm = np.array([var_x.get(), var_y.get(), var_z.get()], dtype=float)
        p_m = mm_to_m_vec(p_mm)
        p_m = clamp_vec3(p_m, workspace_limit_m)
        p_goal_shared[:] = p_m

        # orientation deg
        rpy = np.array([var_roll.get(), var_pitch.get(), var_yaw.get()], dtype=float)
        rpy_goal_shared[:] = rpy
        globals()["R_goal_shared"] = rpy_deg_to_R(rpy[0], rpy[1], rpy[2])

        # params
        step_mm = float(var_step_mm.get())
        dq_max = float(var_dqmax.get())
        joint_damping_base = float(var_damp.get())
        null_space_gain = float(var_ns.get())
        keyboard_enable = bool(var_kb.get())

        v_cmd_max_mm_s = float(var_vmax_mm_s.get())
        w_cmd_max_deg = float(var_wmax.get())

def ui_set_target_current():
    """GOAL/CMD를 현재 EE로 동기화해서 튐 방지 (UI도 mm로 갱신)"""
    global last_reach_time
    with lock:
        p = clamp_vec3(latest_p_cur.copy(), workspace_limit_m)
        p_goal_shared[:] = p
        p_cmd_shared[:] = p
        last_reach_time = 0.0
    sync_ui_entries_from_goal()

def ui_home():
    """현재 EE 위치에서 EE는 유지하고, null-space로 관절을 home 쪽으로 당김"""
    global last_reach_time
    with lock:
        p = clamp_vec3(latest_p_cur.copy(), workspace_limit_m)
        p_goal_shared[:] = p
        p_cmd_shared[:] = p
        last_reach_time = 0.0
    sync_ui_entries_from_goal()

def ui_quit():
    root.quit()
    root.destroy()

# ================
# Scenario helpers
# ================
def refresh_waypoint_list():
    list_wp.delete(0, tk.END)
    for i, wp in enumerate(waypoints):
        pmm = wp["p_mm"]; rpy = wp["rpy"]
        list_wp.insert(
            tk.END,
            f"{i:02d} | {wp['name']} | p=({pmm[0]:.1f},{pmm[1]:.1f},{pmm[2]:.1f})mm rpy=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})"
        )

def ui_record_waypoint():
    """Record current goal(target) or current EE as a waypoint. (store in mm/deg)"""
    with lock:
        src = var_record_source.get()
        if src == "ee":
            p_m = latest_p_cur.copy()
            rpy = R_to_rpy_deg(latest_R_cur.copy())
        else:
            p_m = p_goal_shared.copy()
            rpy = rpy_goal_shared.copy()

    p_m = clamp_vec3(p_m, workspace_limit_m)
    p_mm = m_to_mm_vec(p_m)

    waypoints.append({
        "name": f"WP{len(waypoints)}",
        "p_mm": p_mm.astype(float),
        "rpy": np.array(rpy, dtype=float),
    })
    refresh_waypoint_list()

def ui_delete_selected():
    sel = list_wp.curselection()
    if not sel:
        return
    idx = sel[0]
    if 0 <= idx < len(waypoints):
        waypoints.pop(idx)
        refresh_waypoint_list()

def ui_clear_all():
    waypoints.clear()
    refresh_waypoint_list()

def apply_waypoint(idx):
    """Set GOAL from waypoint (stored mm/deg) -> internal m/deg"""
    global last_reach_time
    if idx < 0 or idx >= len(waypoints):
        return
    wp = waypoints[idx]
    p_m = mm_to_m_vec(wp["p_mm"].copy())
    p_m = clamp_vec3(p_m, workspace_limit_m)

    with lock:
        p_goal_shared[:] = p_m
        rpy_goal_shared[:] = wp["rpy"].copy()
        globals()["R_goal_shared"] = rpy_deg_to_R(rpy_goal_shared[0], rpy_goal_shared[1], rpy_goal_shared[2])
        last_reach_time = 0.0
    sync_ui_entries_from_goal()

def ui_goto_selected():
    sel = list_wp.curselection()
    if not sel:
        return
    apply_waypoint(sel[0])

def ui_play():
    global play_enable, play_idx, play_mode, hold_time, pos_tol_mm, rot_tol, last_reach_time
    if len(waypoints) == 0:
        messagebox.showinfo("Scenario", "웨이포인트가 없습니다. Record WP로 먼저 저장하세요.")
        return
    play_mode = var_playmode.get()
    hold_time = float(var_hold.get())
    pos_tol_mm = float(var_postol_mm.get())
    rot_tol = float(var_rottol.get())

    play_enable = True
    play_idx = 0
    last_reach_time = 0.0
    apply_waypoint(play_idx)

def ui_stop():
    global play_enable
    play_enable = False

def ui_save_csv():
    if len(waypoints) == 0:
        return
    path = filedialog.asksaveasfilename(
        title="Save waypoints (mm)",
        defaultextension=".csv",
        filetypes=[("CSV", "*.csv")]
    )
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("name,px_mm,py_mm,pz_mm,roll_deg,pitch_deg,yaw_deg\n")
        for wp in waypoints:
            p = wp["p_mm"]; rpy = wp["rpy"]
            f.write(f"{wp['name']},{p[0]},{p[1]},{p[2]},{rpy[0]},{rpy[1]},{rpy[2]}\n")

def ui_load_csv():
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
            pmm = np.array([safe_float(parts[1]), safe_float(parts[2]), safe_float(parts[3])], dtype=float)
            rpy = np.array([safe_float(parts[4]), safe_float(parts[5]), safe_float(parts[6])], dtype=float)

            # clamp in meters then store back in mm
            p_m = clamp_vec3(mm_to_m_vec(pmm), workspace_limit_m)
            pmm = m_to_mm_vec(p_m)

            loaded.append({"name": name, "p_mm": pmm, "rpy": rpy})
    except Exception as e:
        messagebox.showerror("Load error", str(e))
        return

    waypoints[:] = loaded
    refresh_waypoint_list()

# =============================
# Layout (left: target, right: scenario)
# =============================
main = ttk.Frame(root, padding=10)
main.grid(row=0, column=0, sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
main.columnconfigure(0, weight=1)
main.columnconfigure(1, weight=1)

# ---- LEFT: Target control ----
frm = ttk.LabelFrame(main, text="Target Control (mm / deg)", padding=10)
frm.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
frm.columnconfigure(1, weight=1)

ttk.Label(frm, text="Goal Position [mm]").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

ttk.Label(frm, text="X").grid(row=1, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_x, width=12).grid(row=1, column=1, sticky="w")
ttk.Label(frm, text=f"limit ±{workspace_limit_mm[0]:.0f}").grid(row=1, column=2, sticky="w")

ttk.Label(frm, text="Y").grid(row=2, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_y, width=12).grid(row=2, column=1, sticky="w")
ttk.Label(frm, text=f"limit ±{workspace_limit_mm[1]:.0f}").grid(row=2, column=2, sticky="w")

ttk.Label(frm, text="Z").grid(row=3, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_z, width=12).grid(row=3, column=1, sticky="w")
ttk.Label(frm, text=f"limit ±{workspace_limit_mm[2]:.0f}").grid(row=3, column=2, sticky="w")

ttk.Label(frm, text="Goal Orientation RPY [deg]").grid(row=4, column=0, columnspan=3, sticky="w", pady=(12, 6))

ttk.Label(frm, text="Roll").grid(row=5, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_roll, width=12).grid(row=5, column=1, sticky="w")

ttk.Label(frm, text="Pitch").grid(row=6, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_pitch, width=12).grid(row=6, column=1, sticky="w")

ttk.Label(frm, text="Yaw").grid(row=7, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_yaw, width=12).grid(row=7, column=1, sticky="w")

ttk.Label(frm, text="Control Params").grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 6))

ttk.Label(frm, text="step [mm/key]").grid(row=9, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_step_mm, width=12).grid(row=9, column=1, sticky="w")

ttk.Label(frm, text="dq_max [rad/step]").grid(row=10, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_dqmax, width=12).grid(row=10, column=1, sticky="w")

ttk.Label(frm, text="damping_base").grid(row=11, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_damp, width=12).grid(row=11, column=1, sticky="w")

ttk.Label(frm, text="null_gain").grid(row=12, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_ns, width=12).grid(row=12, column=1, sticky="w")

ttk.Checkbutton(frm, text="Keyboard Enable (WASD/QE)", variable=var_kb).grid(row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))

ttk.Label(frm, text="Linear cmd vmax [mm/s]").grid(row=14, column=0, sticky="w", pady=(10, 0))
ttk.Entry(frm, textvariable=var_vmax_mm_s, width=12).grid(row=14, column=1, sticky="w", pady=(10, 0))

ttk.Label(frm, text="Orient wmax [deg/s]").grid(row=15, column=0, sticky="w")
ttk.Entry(frm, textvariable=var_wmax, width=12).grid(row=15, column=1, sticky="w")

btns = ttk.Frame(frm)
btns.grid(row=16, column=0, columnspan=3, sticky="we", pady=(10, 0))
btns.columnconfigure(0, weight=1)
btns.columnconfigure(1, weight=1)
btns.columnconfigure(2, weight=1)

ttk.Button(btns, text="Apply Goal", command=ui_apply_target).grid(row=0, column=0, sticky="we", padx=(0, 6))
ttk.Button(btns, text="Sync goal/cmd = current EE", command=ui_set_target_current).grid(row=0, column=1, sticky="we", padx=(0, 6))
ttk.Button(btns, text="Home (null-space)", command=ui_home).grid(row=0, column=2, sticky="we")

ttk.Button(frm, text="Quit", command=ui_quit).grid(row=17, column=0, columnspan=3, sticky="we", pady=(8, 0))

lbl_cur.grid(row=18, column=0, columnspan=3, sticky="w", pady=(10, 0))
lbl_goal.grid(row=19, column=0, columnspan=3, sticky="w")
lbl_cmd.grid(row=20, column=0, columnspan=3, sticky="w")
lbl_err.grid(row=21, column=0, columnspan=3, sticky="w")
lbl_play.grid(row=22, column=0, columnspan=3, sticky="w", pady=(6, 0))

# ---- RIGHT: Scenario ----
sc = ttk.LabelFrame(main, text="Scenario (mm / deg)", padding=10)
sc.grid(row=0, column=1, sticky="nsew")
sc.columnconfigure(0, weight=1)
sc.rowconfigure(1, weight=1)

toprow = ttk.Frame(sc)
toprow.grid(row=0, column=0, sticky="we")
ttk.Label(toprow, text="Record source:").grid(row=0, column=0, sticky="w")
ttk.Radiobutton(toprow, text="Goal(Target)", value="target", variable=var_record_source).grid(row=0, column=1, sticky="w", padx=(8, 0))
ttk.Radiobutton(toprow, text="EE(Current)", value="ee", variable=var_record_source).grid(row=0, column=2, sticky="w", padx=(8, 0))

list_wp.grid(row=1, column=0, sticky="nsew", pady=(8, 8))

wp_btns = ttk.Frame(sc)
wp_btns.grid(row=2, column=0, sticky="we")
for i in range(6):
    wp_btns.columnconfigure(i, weight=1)

ttk.Button(wp_btns, text="Record WP", command=ui_record_waypoint).grid(row=0, column=0, sticky="we", padx=(0, 6))
ttk.Button(wp_btns, text="Goto selected", command=ui_goto_selected).grid(row=0, column=1, sticky="we", padx=(0, 6))
ttk.Button(wp_btns, text="Delete", command=ui_delete_selected).grid(row=0, column=2, sticky="we", padx=(0, 6))
ttk.Button(wp_btns, text="Clear", command=ui_clear_all).grid(row=0, column=3, sticky="we", padx=(0, 6))
ttk.Button(wp_btns, text="Save CSV", command=ui_save_csv).grid(row=0, column=4, sticky="we", padx=(0, 6))
ttk.Button(wp_btns, text="Load CSV", command=ui_load_csv).grid(row=0, column=5, sticky="we")

playfrm = ttk.Frame(sc)
playfrm.grid(row=3, column=0, sticky="we", pady=(8, 0))
for i in range(6):
    playfrm.columnconfigure(i, weight=1)

ttk.Label(playfrm, text="Play mode:").grid(row=0, column=0, sticky="w")
ttk.Radiobutton(playfrm, text="Hold(s)", value="hold", variable=var_playmode).grid(row=0, column=1, sticky="w")
ttk.Radiobutton(playfrm, text="Error tol", value="error", variable=var_playmode).grid(row=0, column=2, sticky="w")

ttk.Label(playfrm, text="Hold [s]").grid(row=1, column=0, sticky="w")
ttk.Entry(playfrm, textvariable=var_hold, width=8).grid(row=1, column=1, sticky="w")

ttk.Label(playfrm, text="pos_tol [mm]").grid(row=1, column=2, sticky="w")
ttk.Entry(playfrm, textvariable=var_postol_mm, width=8).grid(row=1, column=3, sticky="w")

ttk.Label(playfrm, text="rot_tol [rad]").grid(row=1, column=4, sticky="w")
ttk.Entry(playfrm, textvariable=var_rottol, width=8).grid(row=1, column=5, sticky="w")

playbtns = ttk.Frame(sc)
playbtns.grid(row=4, column=0, sticky="we", pady=(8, 0))
playbtns.columnconfigure(0, weight=1)
playbtns.columnconfigure(1, weight=1)

ttk.Button(playbtns, text="Play", command=ui_play).grid(row=0, column=0, sticky="we", padx=(0, 6))
ttk.Button(playbtns, text="Stop", command=ui_stop).grid(row=0, column=1, sticky="we")

# =============================
# UI tick
# =============================
def ui_tick():
    with lock:
        pc = latest_p_cur.copy()
        pd_goal = p_goal_shared.copy()
        pd_cmd = p_cmd_shared.copy()
        rpy_goal = rpy_goal_shared.copy()
        rpy_cmd = R_to_rpy_deg(R_cmd_shared.copy())
        pe = play_enable
        idx = play_idx
        nwp = len(waypoints)

    pc_mm = m_to_mm_vec(pc)
    goal_mm = m_to_mm_vec(pd_goal)
    cmd_mm = m_to_mm_vec(pd_cmd)
    err_mm = np.linalg.norm(cmd_mm - pc_mm)

    lbl_cur.config(text=f"EE cur [mm]: (x,y,z)=({pc_mm[0]: .2f}, {pc_mm[1]: .2f}, {pc_mm[2]: .2f})")
    lbl_goal.config(text=f"GOAL [mm]: (x,y,z)=({goal_mm[0]: .2f}, {goal_mm[1]: .2f}, {goal_mm[2]: .2f}) | rpy=({rpy_goal[0]: .1f},{rpy_goal[1]: .1f},{rpy_goal[2]: .1f})")
    lbl_cmd.config(text=f"CMD  [mm]: (x,y,z)=({cmd_mm[0]: .2f}, {cmd_mm[1]: .2f}, {cmd_mm[2]: .2f}) | rpy=({rpy_cmd[0]: .1f},{rpy_cmd[1]: .1f},{rpy_cmd[2]: .1f})")
    lbl_err.config(text=f"ERR  : |CMD-EE| = {err_mm:.3f} mm")

    if pe:
        lbl_play.config(text=f"Scenario: PLAYING  idx={idx}/{max(nwp-1,0)}  mode={play_mode}")
    else:
        lbl_play.config(text="Scenario: stopped")

    root.after(100, ui_tick)

root.after(100, ui_tick)

# =============================
# Control loop thread
# =============================
stop_flag = False

def control_loop():
    global q, latest_p_cur, latest_R_cur
    global play_enable, play_idx, last_reach_time
    global R_cmd_shared

    last_print = 0.0

    while not stop_flag:
        # FK
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        oMf_local = data.oMf[ee_id]
        p_cur = oMf_local.translation
        R_cur = oMf_local.rotation

        # publish latest EE
        with lock:
            latest_p_cur = p_cur.copy()
            latest_R_cur = R_cur.copy()

            # read params
            kb = bool(keyboard_enable)

            local_step_m = float(step_mm) * MM_TO_M
            local_dqmax = float(dq_max)
            damp_base = float(joint_damping_base)
            ns_gain = float(null_space_gain)

            local_vmax_m_s = float(v_cmd_max_mm_s) * MM_TO_M
            local_wmax_deg = float(w_cmd_max_deg)

            # scenario snapshot
            pe = bool(play_enable)
            mode = play_mode
            htime = float(hold_time)
            ptol_m = float(pos_tol_mm) * MM_TO_M
            rtol = float(rot_tol)
            idx = int(play_idx)
            nwp = len(waypoints)

            # current goals/commands
            p_goal = p_goal_shared.copy()
            R_goal = globals()["R_goal_shared"].copy()
            p_cmd = p_cmd_shared.copy()
            R_cmd = R_cmd_shared.copy()

        # =============================
        # Scenario playback sets GOAL (not CMD)
        # =============================
        if pe and nwp > 0:
            pos_err_goal = p_goal - p_cur
            rot_err_goal = rot_error_log3(R_goal, R_cur)

            reached = False
            if mode == "hold":
                if last_reach_time == 0.0:
                    last_reach_time = time.time()
                if (time.time() - last_reach_time) >= htime:
                    reached = True
            else:  # error tol
                if np.linalg.norm(pos_err_goal) <= ptol_m and np.linalg.norm(rot_err_goal) <= rtol:
                    if last_reach_time == 0.0:
                        last_reach_time = time.time()
                    if (time.time() - last_reach_time) >= 0.15:
                        reached = True
                else:
                    last_reach_time = 0.0

            if reached:
                idx += 1
                if idx >= nwp:
                    with lock:
                        play_enable = False
                else:
                    with lock:
                        play_idx = idx
                    last_reach_time = 0.0
                    apply_waypoint(idx)

            with lock:
                p_goal = p_goal_shared.copy()
                R_goal = globals()["R_goal_shared"].copy()

        # =============================
        # Keyboard updates GOAL when not playing (step is mm/key)
        # =============================
        with lock:
            pe_now = bool(play_enable)

        if kb and (not pe_now):
            if keyboard.is_pressed("w"): p_goal[0] += local_step_m
            if keyboard.is_pressed("s"): p_goal[0] -= local_step_m
            if keyboard.is_pressed("a"): p_goal[1] += local_step_m
            if keyboard.is_pressed("d"): p_goal[1] -= local_step_m
            if keyboard.is_pressed("q"): p_goal[2] += local_step_m
            if keyboard.is_pressed("e"): p_goal[2] -= local_step_m

            p_goal = clamp_vec3(p_goal, workspace_limit_m)

            with lock:
                p_goal_shared[:] = p_goal

        # =============================
        # LINEAR TRAJECTORY: CMD moves in a straight line toward GOAL
        # =============================
        dp = p_goal - p_cmd
        dist = float(np.linalg.norm(dp))
        max_step = local_vmax_m_s * dt
        if dist <= max_step or dist < 1e-12:
            p_cmd = p_goal.copy()
        else:
            p_cmd = p_cmd + (dp / dist) * max_step
        p_cmd = clamp_vec3(p_cmd, workspace_limit_m)

        # orientation rate-limit (SLERP)
        R_rel = R_goal @ R_cmd.T
        angle = float(np.linalg.norm(pin.log3(R_rel)))  # rad
        wmax = np.deg2rad(local_wmax_deg)               # rad/s
        max_ang = wmax * dt
        if angle <= max_ang or angle < 1e-9:
            R_cmd = R_goal.copy()
        else:
            alpha = max_ang / angle
            R_cmd = slerp_R(R_cmd, R_goal, alpha)

        with lock:
            p_cmd_shared[:] = p_cmd
            R_cmd_shared = R_cmd.copy()

        # =============================
        # IK uses CMD targets
        # =============================
        rot_err = rot_error_log3(R_cmd, R_cur)
        pos_err = p_cmd - p_cur
        err = np.hstack([pos_err, rot_err])

        damping = damp_base * (1.0 + np.linalg.norm(err))

        J6 = pin.computeFrameJacobian(
            model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        J = J6[:, 6:]

        JT = J.T
        JJt = J @ JT

        A = JJt + damping * np.eye(6)
        dq = JT @ np.linalg.solve(A, err)

        N = np.eye(J.shape[1]) - JT @ np.linalg.solve(A, J)
        dq += N @ (ns_gain * (q_home[7:] - q[7:]))

        dq = np.clip(dq, -local_dqmax, local_dqmax)

        q[7:] += dq
        q[7:] = np.clip(q[7:], model.lowerPositionLimit[7:], model.upperPositionLimit[7:])

        viz.display(q)

        now = time.time()
        if now - last_print > 0.2:
            last_print = now
            pmm = m_to_mm_vec(p_cur)
            # cmd error in mm
            cmd_err_mm = np.linalg.norm(m_to_mm_vec(p_cmd - p_cur))
            print(f"EE [mm] X:{pmm[0]: .2f} Y:{pmm[1]: .2f} Z:{pmm[2]: .2f} | |CMD-EE|={cmd_err_mm:.3f} mm")

        time.sleep(dt)

# Start control thread
th = threading.Thread(target=control_loop, daemon=True)
th.start()

# Run UI mainloop
try:
    root.mainloop()
finally:
    stop_flag = True
    time.sleep(0.05)
