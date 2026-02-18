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

        # Collision model
        self.collision_model = pin.buildGeomFromUrdf(
            self.model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[mesh_root]
        )
        self.collision_model.addAllCollisionPairs()
        self._remove_adjacent_collision_pairs()
        self.collision_data = self.collision_model.createData()

        # Separate data buffer for trial IK (avoids clobbering self.data)
        self.trial_data = self.model.createData()

        self.ee_name = ee_name
        self.ee_id = self.model.getFrameId(ee_name)

        self.q = pin.neutral(self.model)
        self.q[2] = 0.5
        self.q[3:7] = R.from_euler("xyz", [0, 0, 0]).as_quat()

        self.q_home = self.q.copy()

        self.forward()
        self.display()

    def _remove_adjacent_collision_pairs(self):
        """Remove collision pairs for directly connected links (always in contact)."""
        pairs_to_remove = []
        for k, pair in enumerate(self.collision_model.collisionPairs):
            go1 = self.collision_model.geometryObjects[pair.first]
            go2 = self.collision_model.geometryObjects[pair.second]
            j1, j2 = go1.parentJoint, go2.parentJoint
            if j1 == j2 or self.model.parents[j2] == j1 or self.model.parents[j1] == j2:
                pairs_to_remove.append(k)
        for k in reversed(pairs_to_remove):
            self.collision_model.removeCollisionPair(self.collision_model.collisionPairs[k])

    def forward(self):
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

    def get_ee_pose(self):
        self.forward()
        oMf = self.data.oMf[self.ee_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def compute_jacobian_joints(self) -> np.ndarray:
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

    # --- Safety checks ---

    def check_self_collision(self, q: np.ndarray = None) -> bool:
        """Returns True if self-collision detected at configuration q."""
        if q is None:
            q = self.q
        pin.computeCollisions(
            self.model, self.data,
            self.collision_model, self.collision_data,
            q, True,  # stop_at_first_collision
        )
        return any(
            self.collision_data.collisionResults[k].isCollision()
            for k in range(len(self.collision_model.collisionPairs))
        )

    def is_reachable(self, p_target: np.ndarray, R_target: np.ndarray,
                     n_steps: int = 30, pos_tol: float = 0.003,
                     rot_tol: float = 0.05) -> tuple:
        """Run trial IK on a cloned q to check reachability.

        Returns (reachable: bool, pos_err_m: float, rot_err_rad: float).
        Does NOT modify self.q.
        """
        q_trial = self.q.copy()
        damping = 1e-3
        dq_max = 0.1

        for _ in range(n_steps):
            pin.forwardKinematics(self.model, self.trial_data, q_trial)
            pin.updateFramePlacements(self.model, self.trial_data)

            oMf = self.trial_data.oMf[self.ee_id]
            pos_err = p_target - oMf.translation
            rot_err = pin.log3(R_target @ oMf.rotation.T)

            pe = float(np.linalg.norm(pos_err))
            re = float(np.linalg.norm(rot_err))
            if pe < pos_tol and re < rot_tol:
                return True, pe, re

            J6 = pin.computeFrameJacobian(
                self.model, self.trial_data, q_trial,
                self.ee_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J = J6[:, 6:]

            err = np.hstack([pos_err, rot_err])
            lam = damping * (1.0 + np.linalg.norm(err))
            A = J @ J.T + lam * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -dq_max, dq_max)

            q_trial[7:] += dq
            q_trial[7:] = np.clip(
                q_trial[7:],
                self.model.lowerPositionLimit[7:],
                self.model.upperPositionLimit[7:],
            )

            # Check collision during trial
            if self.check_self_collision(q_trial):
                return False, pe, re

        # Final error after all steps
        pin.forwardKinematics(self.model, self.trial_data, q_trial)
        pin.updateFramePlacements(self.model, self.trial_data)
        oMf = self.trial_data.oMf[self.ee_id]
        pe = float(np.linalg.norm(p_target - oMf.translation))
        re = float(np.linalg.norm(pin.log3(R_target @ oMf.rotation.T)))
        return (pe < pos_tol and re < rot_tol), pe, re
