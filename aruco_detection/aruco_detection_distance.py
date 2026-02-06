import cv2
import numpy as np

# =========================
# ArUco dictionary mapping
# =========================
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# =========================
# Camera intrinsics (yours)
# =========================
K = np.array([
    [933.15867, 0.0, 657.59],
    [0.0, 933.1586, 400.36993],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

D = np.array([-0.43948, 0.18514, 0.0, 0.0], dtype=np.float64)

# =========================
# Marker configuration
# =========================
ARUCO_TYPE = "DICT_5X5_100"

# CHANGE THIS BASED ON TASK:
# Keyboard markers → 0.02 (2 cm)
# USB slot markers → 0.01 (1 cm)
MARKER_SIZE_METERS = 0.02

# Expected marker IDs around object (can be any order)
EXPECTED_IDS = {0, 1, 2, 3}   # change if competition uses different IDs


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[ARUCO_TYPE])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            ids_flat = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for marker_corners, marker_id in zip(corners, ids_flat):

                if marker_id not in EXPECTED_IDS:
                    continue

                # Pose estimation
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners,
                    MARKER_SIZE_METERS,
                    K,
                    D
                )

                # Distance from camera (Euclidean)
                distance = np.linalg.norm(tvec[0][0])

                # Draw axis
                cv2.drawFrameAxes(
                    frame,
                    K,
                    D,
                    rvec[0],
                    tvec[0],
                    MARKER_SIZE_METERS * 0.5
                )

                # Marker center
                pts = marker_corners.reshape(4, 2)
                cX = int(np.mean(pts[:, 0]))
                cY = int(np.mean(pts[:, 1]))

                cv2.putText(
                    frame,
                    f"ID {marker_id} : {distance:.3f} m",
                    (cX - 40, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                print(f"[INFO] Marker {marker_id} distance: {distance:.4f} m")

        cv2.imshow("ArUco Distance Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
