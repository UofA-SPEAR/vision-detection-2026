import os
import json
import shutil
import sys

from subprocess import PIPE, run
from ultralytics import YOLO

HAMMER_DIR = "obj_hammer"
MALLET_DIR = "obj_mallet"
WATERBOTTLE_DIR = "obj_waterbottle"



def split_dataset(src_image_folder, src_labels_folder, dst_image_folder, dst_label_folder):

    image_files = [
        f for f in os.listdir(src_image_folder)
        if os.path.isfile(os.path.join(src_image_folder, f))
    ]

    split_index = int(0.8 * len(image_files))
    train_images = image_files[:split_index]
    val_images   = image_files[split_index:]

    for split_name, image_list in [("train", train_images), ("val", val_images)]:
        for img in image_list:

            
            img_src = os.path.join(src_image_folder, img)
            img_dst = os.path.join(dst_image_folder, split_name, "images", img)


            
            base_name = os.path.splitext(img)[0]
            label = base_name + ".txt"

            lbl_src = os.path.join(src_labels_folder, label)
            lbl_dst = os.path.join(dst_label_folder, split_name, "labels", label)

            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

def reset_folder(dst_folder):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder, exist_ok=True)


def change_labels(folder_path, new_class_id):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                new_lines.append(line)
                continue
            
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts) + "\n")

        with open(file_path, "w") as f:
            f.writelines(new_lines)

def access_dir(_path):
    label_path = os.path.join(_path, "labels")
    
    if "obj_waterbottle" in label_path:
        change_labels(label_path, 0)

    elif "obj_mallet" in label_path:
        change_labels(label_path, 1)

    else:
        change_labels(label_path, 2)


def start_train():
    with open("train.py") as file:
        exec(file.read())


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(base_path, "dataset")

    train_images_dst = os.path.join(dataset_path, "train", "images")
    train_labels_dst = os.path.join(dataset_path, "train", "labels")
    val_images_dst   = os.path.join(dataset_path, "val", "images")
    val_labels_dst   = os.path.join(dataset_path, "val", "labels")

    os.makedirs(train_images_dst, exist_ok=True)
    os.makedirs(train_labels_dst, exist_ok=True)
    os.makedirs(val_images_dst, exist_ok=True)
    os.makedirs(val_labels_dst, exist_ok=True)

    object_dirs = [
        os.path.join(base_path, "obj_waterbottle"),
        os.path.join(base_path, "obj_mallet"),
        os.path.join(base_path, "obj_hammer"),
    ]

    for obj_path in object_dirs:
        access_dir(obj_path)
       
        split_dataset(
            src_image_folder=os.path.join(obj_path, "images"),
            src_labels_folder=os.path.join(obj_path, "labels"),
            dst_image_folder=os.path.join(dataset_path),
            dst_label_folder=os.path.join(dataset_path),
        )

    start_train()


if __name__ == "__main__":
    main()