import numpy as np
import cv2

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):

	if ids is None or len(corners) == 0:
		return image 
	
	if len(corners) > 0:
        
		ids = ids.flatten()

		for (markerCorner, markerID) in zip(corners, ids):
			corners = markerCorner.reshape((4,2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# find the 4 corners of the aruco marker
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# creates the border
			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

			#creates the center
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
		
		return image

aruco_type = "DICT_5X5_100"
id = 1

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()

# marker = cv2.aruco.generateImageMarker(aruco_dict, 1, 250)

# marker = cv2.copyMakeBorder(
#     marker, 50, 50, 50, 50,
#     cv2.BORDER_CONSTANT, value=255
# )

# for default camera
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# zedxonesrc → ZED X One camera driver (GStreamer source)
# camera-resolution=2 → HD1200 (1920×1200)
# camera-fps=30 → 30 frames per second

gst_pipeline = (
    "zedxonesrc camera-resolution=2 camera-fps=30 ! "
    "videoconvert ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    raise RuntimeError("Could not open ZED X One camera")

print(cv2.getBuildInformation())

while cap.isOpened():
	ret, img = cap.read()

	h, w, _ = img.shape

	width = 1000
	height = int(width*(h/w))
	img = cv2.resize(img, (width, height ), interpolation=cv2.INTER_CUBIC)

	corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters = arucoParams)
	
	detected_markers = aruco_display(corners, ids, rejected, img)
	cv2.imshow("Marker", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release()