import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt") 

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("YOLOv8 live detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)

        # Exit when pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
