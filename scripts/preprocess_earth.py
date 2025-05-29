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
TOTAL_SAMPLES = 1000               # Total number of images to download
VAL_SPLIT = 0.2                 # Percentage for validation set
CLOUD_THRESHOLD = 65            # Minimum cloud coverage percentage to consider
SAVE_ROOT = "../datasets/CloudSen12"  # Destination folder for the dataset
CLASS_ID = 0                    # Single class: cloud

os.makedirs(SAVE_ROOT, exist_ok=True)

# === CREATE TRAIN AND VAL FOLDERS ===
for split in ["train", "val"]:
    os.makedirs(f"{SAVE_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/{split}/labels", exist_ok=True)

# === LOAD DATASET ===
dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")  # Load CloudSEN12 dataset
cloudy_df = dataset[dataset["thick_percentage"] > CLOUD_THRESHOLD]  # Filter only cloudy samples
cloudy_indices = cloudy_df.index.tolist()  # Get list of valid indices
random.seed(42)  # Fix seed for reproducibility
random.shuffle(cloudy_indices)  # Shuffle indices randomly

#cloudy_indices = cloudy_indices[:TOTAL_SAMPLES]
valid_indices = []  # To store final selected indices
for idx in cloudy_indices:
    try:
        s2_label = dataset.read(idx).read(1)  # Try reading the label image
        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)  # Read label mask
        if np.any(np.isin(mask, [1, 2])):  # Check if cloud classes 1 or 2 are present
            valid_indices.append(idx)
        if len(valid_indices) >= TOTAL_SAMPLES:
            break  # Stop if required number of samples is reached
    except Exception as e:
        print(f"Error reading mask for idx {idx}: {e}")  # Handle errors

# Warn if fewer valid samples are found
if len(valid_indices) < TOTAL_SAMPLES:
    print(f"Only {len(valid_indices)} valid images found. Reduce TOTAL_SAMPLES or relax the filter.")

# Split into validation and training indices
val_count = int(TOTAL_SAMPLES * VAL_SPLIT)
val_indices = valid_indices[:val_count]
train_indices = valid_indices[val_count:]

print(f"Train: {len(train_indices)} samples | Val: {len(val_indices)} samples")

# === FUNCTION TO PROCESS A SINGLE SAMPLE ===
def process_sample(idx, split):
    try:
        s2_l1c = dataset.read(idx).read(0)  # Read RGB Sentinel-2 image path
        s2_label = dataset.read(idx).read(1)  # Read label mask path

        with rio.open(s2_l1c) as src:
            if src.count < 4:
                print(f"[{split}] Sample {idx} - Image has fewer than 4 bands, skipped.")
                return
            rgb = src.read([4, 3, 2])  # Extract RGB bands (B4, B3, B2)

        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)  # Read cloud mask

        if not np.any(np.isin(mask, [1, 2])):
            print(f"[{split}] Sample {idx} - No cloud classes found.")
            return

        # Save image
        img = np.transpose(rgb, (1, 2, 0))  # Rearrange channels
        img = np.clip(img / 3000, 0, 1) * 255  # Normalize and scale to 8-bit
        img = Image.fromarray(img.astype(np.uint8))
        img_name = f"img_{idx:05d}.png"
        img.save(f"{SAVE_ROOT}/{split}/images/{img_name}")

        # Generate binary cloud mask
        cloud_mask = np.where(np.isin(mask, [1, 2]), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []
        h, w = cloud_mask.shape
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) != 2 or contour.shape[0] < 3:
                continue  # Skip invalid or small contours
            norm = contour.astype(np.float32)
            norm[:, 0] /= w  # Normalize x coordinates
            norm[:, 1] /= h  # Normalize y coordinates
            flat = norm.flatten()
            line = f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in flat)
            label_lines.append(line)

        # Save YOLO polygon label
        label_name = img_name.replace(".png", ".txt")
        with open(f"{SAVE_ROOT}/{split}/labels/{label_name}", "w") as f:
            f.write("\n".join(label_lines))

    except Exception as e:
        print(f"[{split}] Error with sample {idx}: {e}")  # Handle any errors

# === LOOP THROUGH TRAINING SAMPLES ===
print("Processing training samples...")
for idx in tqdm(train_indices):
    process_sample(idx, "train")

# === LOOP THROUGH VALIDATION SAMPLES ===
print("Processing validation samples...")
for idx in tqdm(val_indices):
    process_sample(idx, "val")
