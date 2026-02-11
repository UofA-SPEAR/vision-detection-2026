from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.yaml")
    model.train(data = "conf.yaml", epochs = 10, device='0')

if __name__ == "__main__":
    main()

