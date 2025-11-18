from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

results = model.predict(source="1", show = True)

print(results)