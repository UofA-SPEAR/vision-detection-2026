from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 model 
    model = YOLO('yolov8n.pt')

    model.train(
        data='data/rover_detection.yaml',      
        epochs=50,             
        imgsz=640,             
        batch=8,               
        name='rover_detection', 
        device=0               
    )

if __name__ == "__main__":
    main()
