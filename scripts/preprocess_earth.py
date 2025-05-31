import os
import tacoreader
import rasterio as rio
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import random

# Parameters (same as your original)
TOTAL_SAMPLES_CLOUDY = 2000
CLEAR_PERCENTAGE = 0.2
TOTAL_SAMPLES_CLEAR = int(TOTAL_SAMPLES_CLOUDY * CLEAR_PERCENTAGE)
VAL_SPLIT = 0.2
CLOUD_THRESHOLD = 55
SAVE_ROOT = "../datasets/CloudSen12"
CLASS_ID = 0

os.makedirs(SAVE_ROOT, exist_ok=True)
for split in ["train", "val"]:
    os.makedirs(f"{SAVE_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/{split}/labels", exist_ok=True)

dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")

# Counters
count_cloudy = 0
count_clear = 0

val_cloudy_count = int(TOTAL_SAMPLES_CLOUDY * VAL_SPLIT)
val_clear_count = int(TOTAL_SAMPLES_CLEAR * VAL_SPLIT)

def process_sample(idx, s2_l1c, s2_label, split):
    try:
        with rio.open(s2_l1c) as src:
            if src.count < 4:
                print(f"[{split}] Sample {idx} has less than 4 bands, skipping.")
                return False
            rgb = src.read([4,3,2])
        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)

        img = np.transpose(rgb, (1,2,0))
        img = np.clip(img / 3000, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img_name = f"img_{idx:05d}.png"
        img.save(f"{SAVE_ROOT}/{split}/images/{img_name}")

        cloud_mask = np.where(np.isin(mask, [1, 2]), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_lines = []
        h, w = cloud_mask.shape
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim != 2 or contour.shape[0] < 3:
                continue
            norm = contour.astype(np.float32)
            norm[:, 0] /= w
            norm[:, 1] /= h
            flat = norm.flatten()
            line = f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in flat)
            label_lines.append(line)

        label_name = img_name.replace(".png", ".txt")
        with open(f"{SAVE_ROOT}/{split}/labels/{label_name}", "w") as f:
            f.write("\n".join(label_lines))

        return True
    except Exception as e:
        print(f"[{split}] Error processing sample {idx}: {e}")
        return False


print("Processing cloudy images...")
cloudy_df = dataset[dataset["thick_percentage"] > CLOUD_THRESHOLD]
cloudy_indices = cloudy_df.index.tolist()
random.seed(42)
random.shuffle(cloudy_indices)

for idx in tqdm(cloudy_indices):
    if count_cloudy >= TOTAL_SAMPLES_CLOUDY:
        break
    try:
        sample = dataset.read(idx)
        s2_l1c = sample.read(0)
        s2_label = sample.read(1)
        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)

        # Check if contains cloud pixels
        if np.any(np.isin(mask, [1, 2])):
            # Decide split
            split = "val" if count_cloudy < val_cloudy_count else "train"
            success = process_sample(idx, s2_l1c, s2_label, split)
            if success:
                count_cloudy += 1
    except Exception as e:
        print(f"Error reading cloudy sample idx {idx}: {e}")

print(f"Collected and processed {count_cloudy} cloudy images.")

print("Processing clear images...")
non_cloudy_df = dataset[dataset["thick_percentage"] < CLOUD_THRESHOLD]
non_cloudy_indices = non_cloudy_df.index.tolist()
random.shuffle(non_cloudy_indices)

for idx in tqdm(non_cloudy_indices):
    if count_clear >= TOTAL_SAMPLES_CLEAR:
        break
    try:
        sample = dataset.read(idx)
        s2_l1c = sample.read(0)
        s2_label = sample.read(1)
        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)

        # Must NOT contain cloud pixels
        if not np.any(np.isin(mask, [1, 2])):
            split = "val" if count_clear < val_clear_count else "train"
            success = process_sample(idx, s2_l1c, s2_label, split)
            if success:
                count_clear += 1
    except Exception as e:
        print(f"Error reading clear sample idx {idx}: {e}")

print(f"Collected and processed {count_clear} clear images.")
