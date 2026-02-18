# config/__init__.py
import numpy as np

# =============================
# URDF & Mesh Paths
# =============================
URDF_PATH = r"/Users/kimhaseo/workspace/study/7dof_urdf/7dof_urdf.urdf"
MESH_ROOT = r"/Users/kimhaseo/workspace/study/7dof_urdf"

# =============================
# EE frame name
# =============================
EE_NAME = "end_effector-v1"

# =============================
# Workspace limit (internal meter)
# =============================
WORKSPACE_LIMIT_M = np.array([0.5, 0.5, 0.5], dtype=float)

# =============================
# Default target orientation (deg)
# =============================
DEFAULT_RPY_DEG = (0.0, 0.0, 0.0)

# =============================
# Control defaults
# =============================
DT = 0.01

# Keyboard step (UI is mm)
STEP_MM = 5.0

# IK
DQ_MAX = 0.05
JOINT_DAMPING_BASE = 1e-4
NULL_SPACE_GAIN = 0.1

# Linear cmd rate-limit (UI is mm/s)
V_CMD_MAX_MM_S = 400.0
W_CMD_MAX_DEG_S = 90.0

# Scenario defaults
HOLD_TIME_S = 1.0
POS_TOL_MM = 3.0
ROT_TOL_RAD = 0.03
