import os
import tacoreader
import rasterio as rio
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import random

#TODO decide how many samples for Earth

# === PARAMETERS ===
TOTAL_SAMPLES_CLOUDY = 2000               # Number of cloudy images to select
CLEAR_PERCENTAGE = 0.2                  # Percentage of clear images relative to cloudy ones
TOTAL_SAMPLES_CLEAR = int(TOTAL_SAMPLES_CLOUDY * CLEAR_PERCENTAGE)
VAL_SPLIT = 0.2                         # Fraction of data to use as validation set
CLOUD_THRESHOLD = 45                   # Threshold to classify an image as cloudy (in % of thick cloud)
SAVE_ROOT = "../datasets/CloudSen12"  # Output directory
CLASS_ID = 0                           # Single class ID for clouds

os.makedirs(SAVE_ROOT, exist_ok=True)

# === CREATE TRAIN AND VAL FOLDERS ===
for split in ["train", "val"]:
    os.makedirs(f"{SAVE_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/{split}/labels", exist_ok=True)

# === LOAD DATASET ===
dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")

# === FIND CLOUDY IMAGES ===
cloudy_df = dataset[dataset["thick_percentage"] > CLOUD_THRESHOLD]
cloudy_indices = cloudy_df.index.tolist()
random.seed(42)
random.shuffle(cloudy_indices)

valid_indices = []
for idx in cloudy_indices:
    try:
        s2_label = dataset.read(idx).read(1)
        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)
        if np.any(np.isin(mask, [1, 2])):  # Must contain cloud or thick cloud
            valid_indices.append(idx)
        if len(valid_indices) >= TOTAL_SAMPLES_CLOUDY:
            break
    except Exception as e:
        print(f"Error reading mask for idx {idx}: {e}")

if len(valid_indices) < TOTAL_SAMPLES_CLOUDY:
    print(f"Only {len(valid_indices)} valid cloudy images found. Reduce TOTAL_SAMPLES_CLOUDY or relax the filter.")

# === FIND CLEAR SKY IMAGES ===
non_cloudy_df = dataset[dataset["thick_percentage"] < CLOUD_THRESHOLD]
non_cloudy_indices = non_cloudy_df.index.tolist()
random.shuffle(non_cloudy_indices)

clear_indices = []
for idx in non_cloudy_indices:
    try:
        s2_label = dataset.read(idx).read(1)
        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)
        if not np.any(np.isin(mask, [1, 2])):  # Must NOT contain cloud
            clear_indices.append(idx)
        if len(clear_indices) >= TOTAL_SAMPLES_CLEAR:
            break
    except Exception as e:
        print(f"Error reading clear sample idx {idx}: {e}")

if len(clear_indices) < TOTAL_SAMPLES_CLEAR:
    print(f"Only {len(clear_indices)} clear sky images found. Reduce CLEAR_PERCENTAGE or CLOUD_THRESHOLD.")

# === SPLIT CLOUDY AND CLEAR INTO TRAIN AND VAL ===
val_count = int(TOTAL_SAMPLES_CLOUDY * VAL_SPLIT)
val_clear_count = int(TOTAL_SAMPLES_CLEAR * VAL_SPLIT)

val_indices = valid_indices[:val_count] + clear_indices[:val_clear_count]
train_indices = valid_indices[val_count:] + clear_indices[val_clear_count:]

print(f"Train: {len(train_indices)} samples | Val: {len(val_indices)} samples")

# === FUNCTION TO PROCESS A SINGLE SAMPLE ===
def process_sample(idx, split):
    try:
        s2_l1c = dataset.read(idx).read(0)    # Image path
        s2_label = dataset.read(idx).read(1)  # Label path

        with rio.open(s2_l1c) as src:
            if src.count < 4:
                print(f"[{split}] Sample {idx} - Image has fewer than 4 bands, skipped.")
                return
            rgb = src.read([4, 3, 2])  # Use bands 4, 3, 2 for RGB

        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)

        # Convert to RGB image
        img = np.transpose(rgb, (1, 2, 0))
        img = np.clip(img / 3000, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img_name = f"img_{idx:05d}.png"
        img.save(f"{SAVE_ROOT}/{split}/images/{img_name}")

        # Create binary mask of cloud (class 1 and 2)
        cloud_mask = np.where(np.isin(mask, [1, 2]), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []
        h, w = cloud_mask.shape
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim != 2 or contour.shape[0] < 3:
                print(f"[{split}] Skipping small/invalid contour in idx {idx}")
                continue

            # Normalize coordinates and convert to YOLO-seg format
            norm = contour.astype(np.float32)
            norm[:, 0] /= w
            norm[:, 1] /= h
            flat = norm.flatten()
            line = f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in flat)
            label_lines.append(line)

        label_name = img_name.replace(".png", ".txt")
        with open(f"{SAVE_ROOT}/{split}/labels/{label_name}", "w") as f:
            f.write("\n".join(label_lines))

    except Exception as e:
        print(f"[{split}] Error with sample {idx}: {e}")

# === LOOP THROUGH TRAINING SAMPLES ===
print("Processing training samples...")
for idx in tqdm(train_indices):
    process_sample(idx, "train")

# === LOOP THROUGH VALIDATION SAMPLES ===
print("Processing validation samples...")
for idx in tqdm(val_indices):
    process_sample(idx, "val")
