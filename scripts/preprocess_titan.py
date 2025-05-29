import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image

#TODO before start the code check if original repo dataset contains only subrepositories labels and images, the other must be deleted
random.seed(42)  # Set seed for reproducibility

# === Function to merge train and test folders into a single 'all' folder ===
def merge_train_test_to_all(base_dir):
    all_images = os.path.join(base_dir, "all", "images")
    all_labels = os.path.join(base_dir, "all", "labels")
    os.makedirs(all_images, exist_ok=True)
    os.makedirs(all_labels, exist_ok=True)

    for split in ["train", "test"]:
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")

        for f in os.listdir(img_dir):
            shutil.copy(os.path.join(img_dir, f), os.path.join(all_images, f))  # Copy all images

        for f in os.listdir(lbl_dir):
            shutil.copy(os.path.join(lbl_dir, f), os.path.join(all_labels, f))  # Copy all labels

# === Function to copy .png images from one folder to another ===
def copy_images(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    png_files = list(Path(source_dir).glob("*.png"))
    for file in png_files:
        target_file = Path(target_dir) / file.name
        shutil.copy(file, target_file)
        print(f"Copied: {file} → {target_file}")

# === Function to convert LabelMe JSON labels to YOLO segmentation format ===
def convert_labelme_to_yolo_seg(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = list(Path(json_dir).glob("*.json"))

    for json_file in json_files:
        print(f"Processing {json_file.name}")

        with open(json_file, 'r') as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        yolo_lines = []

        for shape in shapes:
            print(f"  shape_type: {shape.get('shape_type')}  | label: {shape.get('label')}")

            if shape["shape_type"] != "polygon":
                continue  # Skip non-polygon shapes

            points = shape["points"]
            # Normalize coordinates based on image size (1024 or 512)
            if data.get("imageHeight", 1024) == 1024:
                normalized = [coord / 1024 for pt in points for coord in pt]
            else:
                normalized = [coord / 512 for pt in points for coord in pt]

            line = "0 " + " ".join(map(str, normalized))  # class 0 = "cloud"
            yolo_lines.append(line)

        out_path = Path(output_dir) / (json_file.stem + ".txt")
        with open(out_path, 'w') as f:
            f.write("\n".join(yolo_lines))
        # Delete the original .json file after conversion
        os.remove(json_file)

    print(f"Converted {len(json_files)} files to YOLO format at {output_dir}")

# === Function to remove corrupted images and those without labels ===
def clean_dataset(images_dir, labels_dir):
    corrupted_files = []
    missing_labels = []

    for fname in os.listdir(images_dir):
        if not fname.endswith(".png"):
            continue

        path_img = os.path.join(images_dir, fname)
        path_lbl = os.path.join(labels_dir, fname.replace(".png", ".txt"))

        try:
            Image.open(path_img).verify()  # Check for corrupted image
        except Exception:
            print(f"Corrupted image: {path_img}")
            corrupted_files.append(fname)
            os.remove(path_img)
            if os.path.exists(path_lbl):
                os.remove(path_lbl)
            continue

        if not os.path.exists(path_lbl):  # Check for missing label
            print(f"Missing label for {fname}")
            missing_labels.append(fname)
            os.remove(path_img)

    print(f"\n[Clean Summary for {images_dir}]")
    print(f"Corrupted images removed: {len(corrupted_files)}")
    if corrupted_files:
        print(" -", "\n - ".join(corrupted_files))

    print(f"Images with missing labels removed: {len(missing_labels)}")
    if missing_labels:
        print(" -", "\n - ".join(missing_labels))

# === Function to split dataset into train, val, and test ===
def split_dataset(images_dir, labels_dir, output_base):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
    random.seed(42)
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)

    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train+n_val],
        'test': image_files[n_train+n_val:]
    }

    for split, files in splits.items():
        split_img_dir = os.path.join(output_base, split, "images")
        split_lbl_dir = os.path.join(output_base, split, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for fname in files:
            src_img = os.path.join(images_dir, fname)
            src_lbl = os.path.join(labels_dir, fname.replace(".png", ".json"))
            # Copy image only (label will be converted later)
            if os.path.exists(src_img):
                dst_img = os.path.join(split_img_dir, fname)
                shutil.copy(src_img, dst_img)

            # Temporarily copy the .json label (will be converted and deleted)
            if os.path.exists(src_lbl):
                dst_lbl = os.path.join(split_lbl_dir, os.path.basename(src_lbl))
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"[WARNING] No label found for {fname}, image was still copied.")

def merge_train_val_to_full_train(dataset_root):
    """
    Merges train and val images/labels into full_train directory.
    Args:
        dataset_root (str): Path to dataset root containing 'train/' and 'val/' folders.
    """
    full_train_img_dir = os.path.join(dataset_root, 'full_train', 'images')
    full_train_lbl_dir = os.path.join(dataset_root, 'full_train', 'labels')
    os.makedirs(full_train_img_dir, exist_ok=True)
    os.makedirs(full_train_lbl_dir, exist_ok=True)

    for split in ['train', 'val']:
        split_img_dir = os.path.join(dataset_root, split, 'images')
        split_lbl_dir = os.path.join(dataset_root, split, 'labels')

        for img_file in os.listdir(split_img_dir):
            shutil.copy(os.path.join(split_img_dir, img_file), full_train_img_dir)
        for lbl_file in os.listdir(split_lbl_dir):
            shutil.copy(os.path.join(split_lbl_dir, lbl_file), full_train_lbl_dir)

    print(f"✅ Merged train and val into: {full_train_img_dir} and {full_train_lbl_dir}")

# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":
    merge_train_test_to_all("../datasets/Dataset_NASA")

    # 1. Split the dataset
    split_dataset(
        images_dir="../datasets/Dataset_NASA/all/images",
        labels_dir="../datasets/Dataset_NASA/all/labels",
        output_base="../datasets/Titan"
    )

    # 2. Convert LabelMe annotations to YOLO-Seg format for each split
    for split in ["train", "val", "test"]:
        convert_labelme_to_yolo_seg(
            f"../datasets/Titan/{split}/labels",
            f"../datasets/Titan/{split}/labels"
        )

    # 3. Clean the dataset by removing broken images or missing labels
    for split in ["train", "val", "test"]:
        clean_dataset(
            f"../datasets/Titan/{split}/images",
            f"../datasets/Titan/{split}/labels"
        )

    merge_train_val_to_full_train('../datasets/Titan')
