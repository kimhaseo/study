# robot_model.py
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation as R

class RobotModel:
    def __init__(self, urdf_path: str, mesh_root: str, ee_name: str):
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        visual_model = pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.VISUAL, package_dirs=[mesh_root]
        )
        self.viz = MeshcatVisualizer(self.model, None, visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()

        self.ee_name = ee_name
        self.ee_id = self.model.getFrameId(ee_name)

        self.q = pin.neutral(self.model)
        self.q[2] = 0.5
        self.q[3:7] = R.from_euler("xyz", [0, 0, 0]).as_quat()

        self.q_home = self.q.copy()

        self.forward()
        self.p_cur = self.data.oMf[self.ee_id].translation.copy()
        self.R_cur = self.data.oMf[self.ee_id].rotation.copy()

    def forward(self):
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def get_ee_pose(self):
        self.forward()
        oMf = self.data.oMf[self.ee_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def compute_jacobian_joints(self) -> np.ndarray:
        # 6x(nv) then remove free-flyer 6 cols
        J6 = pin.computeFrameJacobian(
            self.model, self.data, self.q, self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J6[:, 6:]

    def apply_dq(self, dq: np.ndarray, dq_max: float):
        dq = np.clip(dq, -dq_max, dq_max)
        self.q[7:] += dq
        self.q[7:] = np.clip(
            self.q[7:], self.model.lowerPositionLimit[7:], self.model.upperPositionLimit[7:]
        )

    def display(self):
        self.viz.display(self.q)
