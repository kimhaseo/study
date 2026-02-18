# camera.py
# D405 + YOLOv8 + PointCloud PCA -> 6DOF pose (x, y, z, roll, pitch, yaw)

import numpy as np
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R


# ============================================================
# D405Detector
# ============================================================
class D405Detector:
    """
    사용법:
        detector = D405Detector("yolov8n.pt")
        pose = detector.get_pose("cup")
        # pose = {"position": np.array([x, y, z]),   # meters, 카메라 좌표
        #         "rpy":      np.array([r, p, y])}    # radians
        detector.close()
    """

    WIDTH  = 1280
    HEIGHT = 720
    FPS    = 30

    # 포인트클라우드에서 유효 depth 범위 (m)
    DEPTH_MIN = 0.1
    DEPTH_MAX = 1.5

    def __init__(self, yolo_model_path: str = "yolov8n.pt"):
        self._init_camera()
        self.model = YOLO(yolo_model_path)

    # ----------------------------------------------------------
    # 카메라 초기화
    # ----------------------------------------------------------
    def _init_camera(self):
        self.pipeline  = rs.pipeline()
        config         = rs.config()
        config.enable_stream(rs.stream.color, self.WIDTH, self.HEIGHT, rs.format.bgr8, self.FPS)
        config.enable_stream(rs.stream.depth, self.WIDTH, self.HEIGHT, rs.format.z16,  self.FPS)

        profile              = self.pipeline.start(config)
        self.align           = rs.align(rs.stream.color)
        color_profile        = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics      = color_profile.get_intrinsics()
        self.pc              = rs.pointcloud()

    # ----------------------------------------------------------
    # 단일 프레임 취득
    # ----------------------------------------------------------
    def _get_frames(self):
        frames       = self.pipeline.wait_for_frames()
        frames       = self.align.process(frames)
        color_frame  = frames.get_color_frame()
        depth_frame  = frames.get_depth_frame()
        return color_frame, depth_frame

    # ----------------------------------------------------------
    # YOLO 검출 -> bbox 반환
    # ----------------------------------------------------------
    def _detect_bbox(self, img: np.ndarray, target_class: str):
        """
        target_class 이름과 일치하는 bbox 중 confidence 가장 높은 것 반환.
        Returns: (x1, y1, x2, y2) or None
        """
        results = self.model(img, verbose=False)
        best_conf = -1.0
        best_box  = None

        for result in results:
            for box in result.boxes:
                cls_name = self.model.names[int(box.cls)]
                conf     = float(box.conf)
                if cls_name == target_class and conf > best_conf:
                    best_conf = conf
                    best_box  = box.xyxy[0].cpu().numpy().astype(int)

        return best_box  # (x1, y1, x2, y2) or None

    # ----------------------------------------------------------
    # bbox 영역 포인트클라우드 추출
    # ----------------------------------------------------------
    def _get_pointcloud_in_bbox(self, depth_frame, bbox):
        x1, y1, x2, y2 = bbox

        self.pc.map_to(depth_frame)
        points    = self.pc.calculate(depth_frame)
        vertices  = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # 픽셀 인덱스 그리드
        us = np.tile(np.arange(self.WIDTH),  self.HEIGHT)
        vs = np.repeat(np.arange(self.HEIGHT), self.WIDTH)

        mask = (us >= x1) & (us < x2) & (vs >= y1) & (vs < y2)
        pts  = vertices[mask]

        # 유효 depth 필터
        valid = (pts[:, 2] > self.DEPTH_MIN) & (pts[:, 2] < self.DEPTH_MAX)
        pts   = pts[valid]

        return pts  # (N, 3)

    # ----------------------------------------------------------
    # PCA로 6DOF 추정
    # ----------------------------------------------------------
    def _pca_pose(self, pts: np.ndarray):
        """
        포인트클라우드 PCA:
          - principal axis 0: 가장 분산 큰 방향 (yaw)
          - principal axis 1: 두 번째 방향 (pitch)
          - normal (axis 2): 카메라 방향 법선 (approach)
        Returns: position (3,), rpy (3,) in radians
        """
        center  = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - center, full_matrices=False)

        # 행벡터: Vt[0]=주축, Vt[2]=법선
        axis0  = Vt[0]   # 객체 주방향
        normal = Vt[2]   # 표면 법선 (approach 방향)

        # normal이 카메라 방향(+Z)을 향하게
        if normal[2] < 0:
            normal = -normal

        # 회전행렬 구성: z축 = normal, x축 = 주방향
        z_ax = normal
        x_ax = axis0 - np.dot(axis0, z_ax) * z_ax
        x_ax = x_ax / (np.linalg.norm(x_ax) + 1e-9)
        y_ax = np.cross(z_ax, x_ax)

        rot_mat = np.column_stack([x_ax, y_ax, z_ax])
        rpy     = R.from_matrix(rot_mat).as_euler("xyz", degrees=False)

        return center, rpy

    # ----------------------------------------------------------
    # Public API: 단일 추정
    # ----------------------------------------------------------
    def get_pose(self, target_class: str) -> dict | None:
        """
        target_class 객체를 검출해 6DOF pose 반환.
        Returns:
            {
                "position": np.array([x, y, z]),  # 카메라 좌표 (m)
                "rpy":      np.array([r, p, y]),  # 카메라 기준 (rad)
                "bbox":     (x1, y1, x2, y2),
                "conf":     float,
            }
            or None (미검출)
        """
        color_frame, depth_frame = self._get_frames()
        if not color_frame or not depth_frame:
            return None

        img  = np.asanyarray(color_frame.get_data())
        bbox = self._detect_bbox(img, target_class)
        if bbox is None:
            return None

        pts = self._get_pointcloud_in_bbox(depth_frame, bbox)
        if len(pts) < 10:
            return None

        position, rpy = self._pca_pose(pts)

        return {
            "position": position,
            "rpy":      rpy,
            "bbox":     tuple(bbox),
        }

    # ----------------------------------------------------------
    # Public API: 라이브 프리뷰 (ESC 종료)
    # ----------------------------------------------------------
    def run_preview(self, target_class: str):
        cv2.namedWindow("D405 Detector")
        print(f"'{target_class}' 감지 중... ESC로 종료")

        while True:
            color_frame, depth_frame = self._get_frames()
            if not color_frame or not depth_frame:
                continue

            img  = np.asanyarray(color_frame.get_data())
            bbox = self._detect_bbox(img, target_class)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                pts = self._get_pointcloud_in_bbox(depth_frame, bbox)

                if len(pts) >= 10:
                    position, rpy = self._pca_pose(pts)
                    x, y, z       = position
                    ro, pi, ya    = np.degrees(rpy)

                    # 시각화
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, target_class,
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img,
                                f"xyz: ({x*1000:.1f}, {y*1000:.1f}, {z*1000:.1f}) mm",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(img,
                                f"rpy: ({ro:.1f}, {pi:.1f}, {ya:.1f}) deg",
                                (x1, y2 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            else:
                cv2.putText(img, f"'{target_class}' not detected",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(img, "ESC to quit",
                        (10, self.HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("D405 Detector", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

    # ----------------------------------------------------------
    def close(self):
        self.pipeline.stop()


# ============================================================
# 단독 실행 테스트
# ============================================================
if __name__ == "__main__":
    detector = D405Detector("yolov8n.pt")
    try:
        detector.run_preview("cup")   # 원하는 클래스로 변경
    finally:
        detector.close()
