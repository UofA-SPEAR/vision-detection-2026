import os
from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor

def get_latest_best_pt(runs_dir="runs/detect"):
    train_dirs = [
        d for d in os.listdir(runs_dir)
        if d.startswith("train") and os.path.isdir(os.path.join(runs_dir, d))
    ]

    if not train_dirs:
        raise FileNotFoundError("No train runs found")

    train_dirs.sort(
        key=lambda x: int(x.replace("train", "")) if x != "train" else 0
    )

    latest_train = train_dirs[-1]
    best_pt = os.path.join(runs_dir, latest_train, "weights", "best.pt")

    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"best.pt not found in {latest_train}")

    return best_pt

def main():

    model_path = get_latest_best_pt()

    model = YOLO(model_path)

    results = model.predict(source="0", show = True)

    print(results)  


if __name__ == "__main__":
    main()