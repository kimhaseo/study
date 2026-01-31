# -*- coding: utf-8 -*-
"""
Pinocchio + Meshcat IK (WASD) + Self-collision "pre-avoid"
+ ✅ 특이점(near-singularity)에서 "날라감" 최소 수정 버전
+ ✅ NEW(요청): 특이점 풀릴 때(=gate 올라갈 때) "확 가속" 방지
    - s_gate release(증가)만 느리게: hysteresis low-pass
    - ddq_max_local도 증가할 때만 천천히 풀리게(rate-limit)

조작:
- WASD: XY, Q/E: Z
- Shift: fast
- ESC: quit
"""

import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R
import keyboard

URDF_PATH = r"C:\Users\kimha\PycharmProjects\study\test_arm\test_arm.urdf"
MESH_ROOT = r"C:\Users\kimha\PycharmProjects\study\test_arm"

# =============================
# Model (fixed base)
# =============================
model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

visual_model = pin.buildGeomFromUrdf(model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT])
viz = MeshcatVisualizer(model, None, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# =============================
# Collision model
# =============================
collision_model = pin.buildGeomFromUrdf(model, URDF_PATH, pin.GeometryType.COLLISION, package_dirs=[MESH_ROOT])
collision_model.addAllCollisionPairs()

def _geom_parent_joint_id(go: pin.GeometryObject) -> int:
    fid = int(go.parentFrame)
    return int(model.frames[fid].parentJoint)

def _auto_filter_collision_pairs(cm: pin.GeometryModel):
    parents = model.parents
    kept = []
    for pair in cm.collisionPairs:
        go1 = cm.geometryObjects[pair.first]
        go2 = cm.geometryObjects[pair.second]
        j1 = _geom_parent_joint_id(go1)
        j2 = _geom_parent_joint_id(go2)

        # same link
        if j1 == j2:
            continue
        # adjacent links
        if parents[j1] == j2 or parents[j2] == j1:
            continue
        kept.append(pair)

    try:
        cm.removeAllCollisionPairs()
    except Exception:
        cm.collisionPairs = []
    for p in kept:
        cm.addCollisionPair(p)

_auto_filter_collision_pairs(collision_model)
collision_data = pin.GeometryData(collision_model)

print(f"[Model] nq={model.nq} nv={model.nv} joints={model.njoints-1}")
print(f"[Collision] geomObjects={len(collision_model.geometryObjects)}, pairs(after filter)={len(collision_model.collisionPairs)}")

# =============================
# Collision check
# =============================
def _update_geom(q_check):
    pin.forwardKinematics(model, data, q_check)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, collision_model, collision_data)

def check_self_collision(q_check):
    try:
        pin.computeCollisions(model, data, collision_model, collision_data, q_check, False)
    except TypeError:
        _update_geom(q_check)
        pin.computeCollisions(collision_model, collision_data, False)

    results = collision_data.collisionResults

    def hit(r):
        if hasattr(r, "isCollision") and callable(getattr(r, "isCollision")):
            try:
                return bool(r.isCollision())
            except Exception:
                pass
        if hasattr(r, "collision"):
            try:
                return bool(r.collision)
            except Exception:
                pass
        if hasattr(r, "numContacts") and callable(getattr(r, "numContacts")):
            try:
                return int(r.numContacts()) > 0
            except Exception:
                pass
        if hasattr(r, "contacts"):
            try:
                return len(r.contacts) > 0
            except Exception:
                pass
        return False

    cnt = 0
    for r in results:
        if hit(r):
            cnt += 1
    return (cnt > 0, cnt)

# =============================
# Utils
# =============================
def ee_orientation_from_euler(roll, pitch, yaw, degrees=True):
    return R.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()

def lowpass(prev, new, a):
    return (1.0 - a) * prev + a * new

def sigma_min(J):
    # 안정적으로: svd singular values로 sigma_min 계산
    s = np.linalg.svd(J, compute_uv=False)
    return float(np.min(s))

def smoothstep01(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

# =============================
# End-effector
# =============================
EE_NAME = "link6"
ee_id = model.getFrameId(EE_NAME)

# =============================
# Control params
# =============================
dt = 0.02

v_key = 0.25
v_key_fast = 0.70

kp_pos = 8.0
kp_rot_nominal = 0.45
kp_rot_min = 0.02
alpha_kprot = 0.25
kp_rot_live = kp_rot_nominal

# --- base damping + singularity adaptive damping ---
damping_base = 5e-4
damping_max = 1e-1

dq_max = 0.11

alpha_p = 0.25
alpha_dq = 0.28

ddq_max = 0.25  # base
null_space_gain = 0.12

workspace_limit = np.array([0.25, 0.25, 0.5], dtype=float)

LOOKAHEAD_STEPS = 10
LOOKAHEAD_GAIN = 2.0

RISK_LA_STEPS = 16
RISK_LA_GAIN = 3.0

CANDIDATE_SCALES = [1.0, 0.6, 0.35, 0.2]
RAND_SAMPLES = 10
RAND_STD_BASE = 0.05
ELBOW_NUDGE_TRIES = 6
ELBOW_NUDGE_STEP = 0.06
ELBOW_Q_IDX = 2  # try 3 if needed

avoid_cooldown_until = 0.0
COOLDOWN_SEC = 0.10

SIG_SOFT = 0.030
SIG_HARD = 0.010

# =============================
# ### MOD-1: 특이점 댐핑 파라미터(강하게)
# =============================
SIG_EPS = 1e-3      # 너무 작으면 튐
LAMBDA_SING = 6e-3
SIG0_SCALE = 0.020

# =============================
# ### MOD-3: 전체 dq 노름 상한(최후 안전벨트)
# =============================
DQ_NORM_MAX = 0.16  # 너무 튀면 줄여라(0.12~0.22)

# =============================
# ✅ NEW: 특이점 "풀릴 때" 확 가속 방지(해제만 천천히)
# =============================
alpha_gate_relax = 0.005    # gate 증가(특이점 풀림) 필터: 작을수록 더 천천히 풀림
alpha_gate_squeeze = 0.35  # gate 감소(특이점 진입) 필터: 클수록 빨리 눌림
s_gate_f = 1.0             # 필터된 gate 상태

ddq_raise_rate = 0.015     # ddq_max_local이 증가할 때 루프당 최대 증가량
ddq_max_local_prev = ddq_max

# =============================
# Init state
# =============================
q = pin.neutral(model)
q_home = q.copy()

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
oMf = data.oMf[ee_id]
p_des = oMf.translation.copy()
p_des_f = p_des.copy()

R_des = ee_orientation_from_euler(0, 180, 0)

dq_f = np.zeros(model.nv)
dq_prev = np.zeros(model.nv)

print("WASD: XY | Q/E: Z | Shift: fast | ESC: quit")

# =============================
# Lookahead collision predictor
# =============================
def will_collide_lookahead(q0, dq_step, steps=LOOKAHEAD_STEPS, gain=LOOKAHEAD_GAIN):
    c0, _ = check_self_collision(q0)
    if c0:
        return True
    for i in range(1, steps + 1):
        t = i / steps
        alpha = (t * t) * gain
        qf = pin.integrate(model, q0, alpha * dq_step)
        c, _ = check_self_collision(qf)
        if c:
            return True
    return False

def risk_level_from_dq(q_now, dq_nom, dq_max_local):
    c, _ = check_self_collision(q_now)
    if c:
        return 1.0

    multipliers = [0.6, 1.0, 1.6, 2.4, 3.2]
    hit_at = None
    for idx, m in enumerate(multipliers):
        dq_test = np.clip(m * dq_nom, -dq_max_local, dq_max_local)
        if will_collide_lookahead(q_now, dq_test, steps=RISK_LA_STEPS, gain=RISK_LA_GAIN):
            hit_at = idx
            break

    if hit_at is None:
        return 0.0
    risk = 1.0 - (hit_at / (len(multipliers) - 1))
    return float(np.clip(risk, 0.0, 1.0))

# =============================
# Candidate generator
# =============================
def pick_safe_dq(q_now, dq_nom, J, inv_term, dq_max_local, rand_std):
    nv = model.nv
    I = np.eye(nv)

    J_pinv = J.T @ inv_term
    N = I - (J_pinv @ J)

    if time.time() < avoid_cooldown_until:
        return np.zeros_like(dq_nom), True

    # scale candidates
    for s in CANDIDATE_SCALES:
        dq_c = np.clip(s * dq_nom, -dq_max_local, dq_max_local)
        if not will_collide_lookahead(q_now, dq_c):
            return dq_c, True

    # null-space random
    for _ in range(RAND_SAMPLES):
        rnd = np.random.randn(nv) * rand_std
        dq_c = dq_nom + (N @ rnd)
        dq_c = np.clip(dq_c, -dq_max_local, dq_max_local)
        if not will_collide_lookahead(q_now, dq_c):
            return dq_c, True

    # elbow nudge
    for k in range(1, ELBOW_NUDGE_TRIES + 1):
        for sign in (+1.0, -1.0):
            dq_c = 0.4 * dq_nom
            if 0 <= ELBOW_Q_IDX < nv:
                dq_c[ELBOW_Q_IDX] += sign * (ELBOW_NUDGE_STEP * k)
            dq_c = np.clip(dq_c, -dq_max_local, dq_max_local)
            if not will_collide_lookahead(q_now, dq_c):
                return dq_c, True

    return np.zeros_like(dq_nom), False

# =============================
# Main loop
# =============================
_last_dbg = 0.0

while True:
    if keyboard.is_pressed("esc"):
        break

    # FK
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[ee_id]
    p_cur = oMf.translation
    R_cur = oMf.rotation

    # input
    v = v_key_fast if keyboard.is_pressed("shift") else v_key
    if keyboard.is_pressed("w"): p_des[0] += v * dt
    if keyboard.is_pressed("s"): p_des[0] -= v * dt
    if keyboard.is_pressed("a"): p_des[1] += v * dt
    if keyboard.is_pressed("d"): p_des[1] -= v * dt
    if keyboard.is_pressed("q"): p_des[2] += v * dt
    if keyboard.is_pressed("e"): p_des[2] -= v * dt
    p_des = np.clip(p_des, -workspace_limit, workspace_limit)

    p_des_f = lowpass(p_des_f, p_des, alpha_p)
    p_des_use = p_des_f

    pos_err = p_des_use - p_cur
    rot_err = pin.log3(R_des @ R_cur.T)

    # Jacobian
    J = pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # singularity measure
    sig = sigma_min(J)

    # gate (raw)
    x = (sig - SIG_HARD) / max(SIG_SOFT - SIG_HARD, 1e-9)
    s_gate = smoothstep01(x)  # 0(hard)~1(soft)

    # =============================
    # ✅ NEW: gate release(증가)는 천천히, squeeze(감소)는 빠르게
    # =============================
    a_gate = alpha_gate_relax if s_gate > s_gate_f else alpha_gate_squeeze
    s_gate_f = (1.0 - a_gate) * s_gate_f + a_gate * s_gate
    s_gate = s_gate_f

    # =============================
    # MOD-1: damping 강하게 sig로 증가
    # =============================
    damping_sing = (LAMBDA_SING / (sig + SIG_EPS)) ** 2
    damping = damping_base + damping_sing
    damping = float(np.clip(damping, damping_base, damping_max))

    # dq_max down near singular
    dq_max_local = dq_max * (0.35 + 0.65 * s_gate)

    # rand std down near singular
    rand_std = RAND_STD_BASE * (0.25 + 0.75 * s_gate)

    # ---- risk relax ----
    err6_nom = np.hstack([kp_pos * pos_err, kp_rot_nominal * rot_err])
    JJt = J @ J.T
    inv_term = np.linalg.inv(JJt + damping * np.eye(6))

    dq_task_nom = J.T @ inv_term @ err6_nom

    I = np.eye(model.nv)
    J_pinv = J.T @ inv_term
    N = I - (J_pinv @ J)
    dq_null = N @ (null_space_gain * (q_home - q))

    dq_nom_tmp = dq_task_nom + dq_null
    if not np.all(np.isfinite(dq_nom_tmp)):
        dq_nom_tmp[:] = 0.0
    dq_nom_tmp = np.clip(dq_nom_tmp, -dq_max_local, dq_max_local)

    risk = risk_level_from_dq(q, dq_nom_tmp, dq_max_local)
    kp_rot_target = kp_rot_min + (kp_rot_nominal - kp_rot_min) * (1.0 - risk)
    kp_rot_live = lowpass(kp_rot_live, kp_rot_target, alpha_kprot)

    # singularity에서 자세도 추가로 풀기
    kp_rot_live_eff = kp_rot_live * (0.30 + 0.70 * s_gate)

    # ---- final dq recompute ----
    err6 = np.hstack([kp_pos * pos_err, kp_rot_live_eff * rot_err])
    inv_term = np.linalg.inv(JJt + damping * np.eye(6))

    dq_task = J.T @ inv_term @ err6
    J_pinv = J.T @ inv_term
    N = I - (J_pinv @ J)
    dq_null = N @ (null_space_gain * (q_home - q))

    dq_nom = dq_task + dq_null
    if not np.all(np.isfinite(dq_nom)):
        dq_nom[:] = 0.0

    # =============================
    # MOD-2: 특이점 dq 스케일링(폭발 억제)
    # =============================
    sing_scale = sig / (sig + SIG0_SCALE)  # sig 작아질수록 0으로
    dq_nom = dq_nom * sing_scale

    dq_nom = np.clip(dq_nom, -dq_max_local, dq_max_local)

    # sampling avoidance
    dq_sel, ok = pick_safe_dq(q, dq_nom, J, inv_term, dq_max_local, rand_std)

    # dq lowpass
    dq_f = lowpass(dq_f, dq_sel, alpha_dq)

    # =============================
    # MOD-3: ddq_max를 특이점에서 자동 감소 + dq norm 상한
    # + ✅ NEW: ddq_max_local "증가만" 천천히 풀기 (특이점 풀릴 때 급가속 방지)
    # =============================
    ddq_target = ddq_max * (0.25 + 0.75 * s_gate)

    if ddq_target > ddq_max_local_prev:
        ddq_max_local = ddq_max_local_prev + min(ddq_raise_rate, ddq_target - ddq_max_local_prev)
    else:
        ddq_max_local = ddq_target
    ddq_max_local_prev = ddq_max_local

    ddq = np.clip(dq_f - dq_prev, -ddq_max_local, ddq_max_local)
    dq_apply = dq_prev + ddq

    # 전체 dq 크기 안전벨트
    nrm = float(np.linalg.norm(dq_apply))
    dq_norm_max_local = DQ_NORM_MAX * (0.35 + 0.65 * s_gate)
    if nrm > dq_norm_max_local and nrm > 1e-12:
        dq_apply *= (dq_norm_max_local / nrm)

    dq_prev = dq_apply.copy()

    # pre-check
    if will_collide_lookahead(q, dq_apply):
        dq_apply[:] = 0.0
        ok = False

    q_next = pin.integrate(model, q, dq_apply)

    # post-check
    c_after, _ = check_self_collision(q_next)
    if c_after:
        q_next = q
        ok = False

    if not ok:
        avoid_cooldown_until = time.time() + COOLDOWN_SEC
        # 목표 리셋
        pin.forwardKinematics(model, data, q_next)
        pin.updateFramePlacements(model, data)
        p_des = data.oMf[ee_id].translation.copy()
        p_des_f = p_des.copy()

    q = q_next
    viz.display(q)

    now = time.time()
    if now - _last_dbg > 1.0:
        _last_dbg = now
        c, n = check_self_collision(q)
        print(
            f"[dbg] ok={ok} sig={sig:.5f} gate={s_gate:.2f} "
            f"damp={damping:.4e} sing_scale={sing_scale:.2f} "
            f"dqmax={dq_max_local:.3f} ddqmax={ddq_max_local:.3f} "
            f"risk={risk:.2f} kp_rot={kp_rot_live_eff:.3f} "
            f"collision={c} pairs={n} |dq|={np.linalg.norm(dq_apply):.3f}"
        )

    time.sleep(dt)

print("Bye.")
