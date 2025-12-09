#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import cv2
from cv2 import aruco
import numpy as np


class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_pose_node')

        # Adjust these topic names to match your ZED-X wrapper
        image_topic = '/zedx/zed_node/left/image_rect_color'
        cam_info_topic = '/zedx/zed_node/left/camera_info'

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # We only care about ArUco IDs 0, 1, 2
        self.target_ids = {0, 1, 2}

        # ArUco dictionary & detector params (4x4_50 as in your spec)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        # Marker size: 0.20 m
        self.marker_length = 0.20

        # Subscriptions
        self.cam_info_sub = self.create_subscription(
            CameraInfo, cam_info_topic, self.camera_info_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        # Optional: publisher of pose (distance etc.)
        self.pose_pub = self.create_publisher(
            PoseStamped, 'aruco_pose', 10
        )

        self.get_logger().info('ArucoPoseNode started.')

    # --- Callbacks ---

    def camera_info_callback(self, msg: CameraInfo):
        """Grab camera intrinsics from CameraInfo (only once)."""
        if self.camera_matrix is not None:
            return

        k = np.array(msg.k).reshape(3, 3)
        d = np.array(msg.d)

        self.camera_matrix = k
        self.dist_coeffs = d

        self.get_logger().info('Camera intrinsics received.')

    def image_callback(self, msg: Image):
        """Main detection + pose estimation."""
        if self.camera_matrix is None:
            # Wait until we have camera intrinsics
            return

        # ROS Image -> OpenCV BGR image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None:
            return

        ids = ids.flatten()
        for corner, marker_id in zip(corners, ids):
            if marker_id not in self.target_ids:
                continue

            # Pose estimation for this marker
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corner,
                self.marker_length,
                self.camera_matrix,
                self.dist_coeffs
            )

            # rvecs/tvecs have shape (1,1,3) for a single marker
            rvec = rvecs[0, 0, :]
            tvec = tvecs[0, 0, :]   # [tx, ty, tz] in meters, camera frame

            distance = float(np.linalg.norm(tvec))

            # Log distances
            self.get_logger().info(
                f'Marker {marker_id}: '
                f'x={tvec[0]:.2f} m, y={tvec[1]:.2f} m, z={tvec[2]:.2f} m, '
                f'distance={distance:.2f} m'
            )

            # Publish pose (optional, in camera frame)
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.pose.position.x = tvec[0]
            pose_msg.pose.position.y = tvec[1]
            pose_msg.pose.position.z = tvec[2]
            # Orientation from rotation vector
            R, _ = cv2.Rodrigues(rvec)
            # Convert R to quaternion
            q = self.rotation_matrix_to_quaternion(R)
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]

            self.pose_pub.publish(pose_msg)

    # --- Helpers ---

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray):
        """Convert 3x3 rotation matrix to (x, y, z, w) quaternion."""
        # Adapted from standard formulas
        q = np.empty(4, dtype=np.float64)
        trace = np.trace(R)

        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            q[3] = 0.25 * s
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = (R[0, 2] - R[2, 0]) / s
            q[2] = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s

        return q  # (x, y, z, w)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

