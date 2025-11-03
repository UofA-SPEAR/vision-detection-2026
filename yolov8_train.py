from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 model (small version for speed)
    model = YOLO('yolov8n.pt')

    # Train the model on your dataset
    model.train(
        data='data/rover_detection.yaml',      # dataset config
        epochs=50,             # you can adjust this
        imgsz=640,             # image size
        batch=8,               # adjust based on GPU memory
        name='rover_detection',  # experiment name
        device=0               # use GPU (or 'cpu' if no CUDA)
    )

if __name__ == "__main__":
    main()
