# -*- coding: utf-8 -*-
"""
Pinocchio IK(ArUco/RealSense) -> Joint angles(q[7:]) rad -> deg -> CAN MotorController

- Pinocchio: q[7:] = joint angles (rad)
- MotorController: AngleCommand(angle=deg, speed=deg/s)
- Includes:
  * RealSense + ArUco pose tracking
  * deadzone for position/orientation jitter
  * 6D IK (pos+rot) with damping
  * CAN send (A4 angle control) using your MotorController/AngleCommand

주의:
- URDF joint 개수가 6개가 아닐 수 있음. 아래 JOINT_NAMES 개수와 model.nq 확인해서 맞춰야 함.
- 방향(sign), 오프셋(offset)은 실제 로봇에 맞게 튜닝 필요.
"""

import sys
import os
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import cv2

# =========================
# ✅ your project imports
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from can_handler import CanHandler
from config.motor_cmd import AngleCommand


# ============================================================
# MotorController (your code 그대로, 일부 출력 문구만 정리)
# ============================================================
class MotorController:
    def __init__(self):
        self.can_handler = CanHandler()

    def move_motor_to_angle(self, angle_command):
        try:
            angle = angle_command.angle         # degree
            speed = angle_command.speed         # deg/s (assumed)
            can_id = angle_command.can_id

            angle_control = int(angle * 1000)   # 0.001 deg unit

            command_byte = 0xA4
            null_byte = 0x00

            speed_limit_low = speed & 0xFF
            speed_limit_high = (speed >> 8) & 0xFF

            angle_control_low = angle_control & 0xFF
            angle_control_mid1 = (angle_control >> 8) & 0xFF
            angle_control_mid2 = (angle_control >> 16) & 0xFF
            angle_control_high = (angle_control >> 24) & 0xFF

            data = [
                command_byte,
                null_byte,
                speed_limit_low,
                speed_limit_high,
                angle_control_low,
                angle_control_mid1,
                angle_control_mid2,
                angle_control_high
            ]

            self.can_handler.send_message(can_id, data)
            print(f"[CAN] {angle_command.motor_name}: {angle:.2f} deg @ {speed} (deg/s)")

        except Exception as e:
            print(f"Error moving motor {angle_command.motor_name}: {e}")
            raise

    def move_motors_to_angle(self, commands: list[AngleCommand]):
        for cmd in commands:
            self.move_motor_to_angle(cmd)

    def close(self):
        self.can_handler.close()


# ============================================================
# ✅ User-editable: paths / joint mapping / motor mapping
# ============================================================
URDF_PATH = r"C:\Users\gktj0\PycharmProjects\study\test_arm\test_arm.urdf"
MESH_ROOT = r"C:\Users\gktj0\PycharmProjects\study\test_arm"

EE_NAME = "link6"

# 로봇팔 관절 이름(전송 순서)
JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
]

# 🔥 실제 모터 방향이 반대면 -1로 바꾸기
JOINT_SIGN = np.array([+1, +1, +1, +1, -1, +1], dtype=float)

# 🔥 홈 오프셋(도 단위). 예: joint2가 기본 90도면 90 넣기
JOINT_OFFSET_DEG = np.array([0, 0, 0, 0, 0, 0], dtype=float)

# CAN 전송 속도 제한 (deg/s)
DEFAULT_SPEED_DPS = 360


# ============================================================
# IK control params
# ============================================================
dt = 0.01
dq_max = 0.05
damping = 1e-4

w_pos = 1.0
w_rot = 1.0

# end-effector desired base orientation
R_des_base = R.from_euler('xyz', [0, -90, 0], degrees=True).as_matrix()
R_des = R_des_base.copy()

# workspace clamp (meters)
ws_min = np.array([-0.25, -0.10,  0.10])
ws_max = np.array([ 0.15,  0.25,  0.50])

# Deadzone
POS_DEADZONE_M = 0.003             # 3mm
ANG_DEADZONE_RAD = np.deg2rad(0.8) # 0.8 deg

# Orientation filter gains
roll_filt = 0.0
pitch_filt = 0.0

ROLL_ALPHA = 0.25
PITCH_ALPHA = 0.25

ROLL_GAIN = 2.0
PITCH_GAIN = 2.0

ROLL_MAX_DEG = 60.0
PITCH_MAX_DEG = 60.0


# ============================================================
# Helper functions
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def map_range(v, in_min, in_max, out_min, out_max):
    if abs(in_max - in_min) < 1e-9:
        return 0.5 * (out_min + out_max)
    t = (v - in_min) / (in_max - in_min)
    t = clamp(t, 0.0, 1.0)
    return out_min + t * (out_max - out_min)

def cam_to_robot_raw(cam_xyz):
    cx, cy, cz = cam_xyz
    rx =  cz
    ry = -cx
    rz = -cy
    return np.array([rx, ry, rz], dtype=float)

# camera ranges (meters)
camX_min, camX_max = -0.30, 0.30
camY_min, camY_max = -0.20, 0.20
camZ_min, camZ_max =  0.05, 0.50

def robot_raw_to_pdes(robot_raw):
    rx, ry, rz = robot_raw
    px = map_range(rx, camZ_min, camZ_max, ws_min[0], ws_max[0])
    py = map_range(ry, -camX_max, -camX_min, ws_min[1], ws_max[1])
    pz = map_range(rz, -camY_max, -camY_min, ws_min[2], ws_max[2])
    return np.array([px, py, pz], dtype=float)

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def deadzone_scalar(x, th):
    return 0.0 if abs(x) < th else x

def deadzone_vec(v, th):
    v = np.asarray(v, dtype=float).copy()
    v[np.abs(v) < th] = 0.0
    return v

def q_to_angle_commands(q, speed_dps=DEFAULT_SPEED_DPS):
    """
    Pinocchio q -> [AngleCommand...] (degree)
    q[0:3] = base position (m)
    q[3:7] = base quaternion
    q[7:]  = joints (rad)
    """
    joints_rad = np.asarray(q[7:], dtype=float)

    if len(joints_rad) < len(JOINT_NAMES):
        raise ValueError(
            f"URDF joints in q[7:] are {len(joints_rad)} but JOINT_NAMES={len(JOINT_NAMES)}. "
            f"맞춰줘야 함."
        )

    joints_rad = joints_rad[:len(JOINT_NAMES)]
    joints_deg = np.degrees(joints_rad)

    # apply sign + offset
    joints_deg = JOINT_SIGN * joints_deg + JOINT_OFFSET_DEG

    cmds = []
    for name, deg in zip(JOINT_NAMES, joints_deg):
        cmds.append(AngleCommand(name, float(deg), int(speed_dps)))
    return cmds, joints_deg


# ============================================================
# Main
# ============================================================
def main():
    # -----------------------------
    # Pinocchio + Meshcat
    # -----------------------------
    model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
    data = model.createData()

    visual_model = pin.buildGeomFromUrdf(
        model, URDF_PATH, pin.GeometryType.VISUAL, package_dirs=[MESH_ROOT]
    )
    viz = MeshcatVisualizer(model, None, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    q = pin.neutral(model)
    q[2] = 0.5
    q[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

    ee_id = model.getFrameId(EE_NAME)

    # init p_des from current
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p_des = data.oMf[ee_id].translation.copy()

    # -----------------------------
    # RealSense setup
    # -----------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    K = np.array([[intr.fx, 0.0,     intr.ppx],
                  [0.0,     intr.fy, intr.ppy],
                  [0.0,     0.0,     1.0     ]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    # -----------------------------
    # ArUco setup
    # -----------------------------
    ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        params = cv2.aruco.DetectorParameters()
    except Exception:
        params = cv2.aruco.DetectorParameters_create()

    TARGET_ID = 0
    MARKER_SIZE_M = 0.05

    last_cam_xyz = None
    last_print = 0.0

    # -----------------------------
    # CAN motor controller
    # -----------------------------
    mc = MotorController()

    # ✅ CAN 송신 주기 제한용 (여기!)
    SEND_HZ = 20
    SEND_DT = 1.0 / SEND_HZ
    last_send = 0.0

    print("ArUco -> IK -> q(rad) -> deg -> CAN send")
    print("ESC on OpenCV window to quit")

    global roll_filt, pitch_filt, R_des

    try:
        while True:
            frames = pipeline.poll_for_frames()
            if not frames:
                time.sleep(0.001)
                continue
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=params)

            got_pose = False
            rvec = None

            if ids is not None and len(ids) > 0:
                ids_flat = ids.flatten()
                if TARGET_ID in ids_flat:
                    idx = int(np.where(ids_flat == TARGET_ID)[0][0])
                    c = corners[idx]

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        c, MARKER_SIZE_M, K, dist
                    )
                    rvec = rvecs[0][0]
                    tvec = tvecs[0][0]
                    got_pose = True

                    last_cam_xyz = np.array([tvec[0], tvec[1], tvec[2]], dtype=float)

                    cv2.aruco.drawDetectedMarkers(color_img, corners, ids)
                    cv2.drawFrameAxes(color_img, K, dist, rvec, tvec, MARKER_SIZE_M * 0.5)

            # ---- position update (deadzone on dp) ----
            if last_cam_xyz is not None:
                robot_raw = cam_to_robot_raw(last_cam_xyz)
                p_mapped = robot_raw_to_pdes(robot_raw)
                p_mapped = np.minimum(np.maximum(p_mapped, ws_min), ws_max)

                dp = p_mapped - p_des
                dp = deadzone_vec(dp, POS_DEADZONE_M)
                p_des = p_des + dp

            # ---- orientation update (deadzone on angular error) ----
            if got_pose and rvec is not None:
                R_cm, _ = cv2.Rodrigues(rvec)
                eul = R.from_matrix(R_cm).as_euler('xyz', degrees=False)

                pitch = eul[0]
                roll = eul[2]

                err_r = deadzone_scalar(wrap_pi(roll - roll_filt), ANG_DEADZONE_RAD)
                err_p = deadzone_scalar(wrap_pi(pitch - pitch_filt), ANG_DEADZONE_RAD)

                roll_filt = wrap_pi(roll_filt + ROLL_ALPHA * err_r)
                pitch_filt = wrap_pi(pitch_filt + PITCH_ALPHA * err_p)

                pitch_cmd = wrap_pi(PITCH_GAIN * pitch_filt)
                roll_cmd = wrap_pi(ROLL_GAIN * roll_filt)

                pitch_lim = np.deg2rad(PITCH_MAX_DEG)
                roll_lim = np.deg2rad(ROLL_MAX_DEG)
                pitch_cmd = float(np.clip(pitch_cmd, -pitch_lim, pitch_lim))
                roll_cmd = float(np.clip(roll_cmd, -roll_lim, roll_lim))

                R_pitch = R.from_euler('y', pitch_cmd, degrees=False).as_matrix()
                R_roll = R.from_euler('z', roll_cmd, degrees=False).as_matrix()
                R_des = (R_roll @ R_pitch) @ R_des_base

            # ---- overlay ----
            if last_cam_xyz is not None:
                cv2.putText(color_img,
                            f"cam[m] x:{last_cam_xyz[0]:.3f} y:{last_cam_xyz[1]:.3f} z:{last_cam_xyz[2]:.3f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(color_img,
                            f"p_des[m] x:{p_des[0]:.3f} y:{p_des[1]:.3f} z:{p_des[2]:.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(color_img, "No marker", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.putText(color_img,
                        f"roll_deg:{np.degrees(roll_filt):.1f}  pitch_deg:{np.degrees(pitch_filt):.1f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(color_img,
                        f"deadzone: pos<{POS_DEADZONE_M*1000:.0f}mm ang<{np.degrees(ANG_DEADZONE_RAD):.1f}deg",
                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow("realsense aruco", color_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # =============================
            # IK (6D)
            # =============================
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[ee_id]
            p_cur = oMf.translation
            R_cur = oMf.rotation

            pos_err = p_des - p_cur
            rot_err = pin.log3(R_des @ R_cur.T)
            err6 = np.hstack([w_pos * pos_err, w_rot * rot_err])

            J6 = pin.computeFrameJacobian(
                model, data, q, ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J = J6[:, 6:]  # base 제외

            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + damping * np.eye(6), err6)
            dq = np.clip(dq, -dq_max, dq_max)

            # integrate
            q[7:] += dq
            q[7:] = np.clip(q[7:], model.lowerPositionLimit[7:], model.upperPositionLimit[7:])

            viz.display(q)

            # =============================
            # ✅ q(rad) -> deg -> CAN send
            # =============================
            try:
                cmds, joints_deg = q_to_angle_commands(q, speed_dps=DEFAULT_SPEED_DPS)

                TEST_JOINT_IDX = 2  # left_joint2

                # ✅ left_joint2면 +60도 보정
                cmds[1].angle += 60.0

                now = time.time()
                if now - last_send > SEND_DT:
                    last_send = now
                    mc.move_motors_to_angle(cmds)


            except Exception as e:
                print(f"[WARN] CAN send skipped: {e}")

            # print debug
            now = time.time()
            if now - last_print > 0.2:
                last_print = now
                if last_cam_xyz is None:
                    print("aruco: none")
                else:
                    jstr = " ".join([f"j{i+1}:{joints_deg[i]:+.1f}°" for i in range(len(JOINT_NAMES))])
                    print(f"p_des=({p_des[0]:+.3f},{p_des[1]:+.3f},{p_des[2]:+.3f})  "
                          f"roll={np.degrees(roll_filt):+.1f} pitch={np.degrees(pitch_filt):+.1f}  |  {jstr}")

            time.sleep(dt)

    finally:
        try:
            mc.close()
        except Exception:
            pass
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
