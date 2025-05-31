import os
import random
import tacoreader
import rasterio as rio
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# === CONFIGURABLE PARAMETERS ===
TOTAL_CLOUDY = 2000                     # Cloudy images (must contain class 1 or 2)
CLEAR_PERCENTAGE = 0.2                  # % of clear sky images
TOTAL_CLEAR = int(TOTAL_CLOUDY * CLEAR_PERCENTAGE)
CLOUD_THRESHOLD = 45                    # Threshold % of thick cloud to consider as "cloudy"
VAL_SPLIT = 0.2                         # Percentage for validation set
SAVE_ROOT = "../datasets/CloudSen12"   # Save directory
CLASS_ID = 0                            # YOLO class

# === CREATE FOLDERS ===
for split in ["train", "val"]:
    os.makedirs(f"{SAVE_ROOT}/{split}/images", exist_ok=True)
    os.makedirs(f"{SAVE_ROOT}/{split}/labels", exist_ok=True)

# === LOAD DATASET ===
dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
random.seed(42)

# === SELECT CLOUDY IMAGES (based on metadata only, not mask) ===
cloudy_df = dataset[dataset["thick_percentage"] > CLOUD_THRESHOLD]
cloudy_indices = cloudy_df.index.tolist()
random.shuffle(cloudy_indices)
cloudy_valid = cloudy_indices[:TOTAL_CLOUDY]

# === SELECT CLEAR SKY IMAGES (based on metadata only, not mask) ===
clear_df = dataset[dataset["thick_percentage"] <= CLOUD_THRESHOLD]
clear_indices = clear_df.index.tolist()
random.shuffle(clear_indices)
clear_valid = clear_indices[:TOTAL_CLEAR]

# === CHECK IF ENOUGH IMAGES WERE FOUND ===
if len(cloudy_valid) < TOTAL_CLOUDY:
    print(f"[!] Only {len(cloudy_valid)} cloudy images found. Reduce TOTAL_CLOUDY or lower CLOUD_THRESHOLD.")
if len(clear_valid) < TOTAL_CLEAR:
    print(f"[!] Only {len(clear_valid)} clear sky images found. Reduce CLEAR_PERCENTAGE or CLOUD_THRESHOLD.")

# === TRAIN / VAL SPLIT ===
val_cloudy = int(len(cloudy_valid) * VAL_SPLIT)
val_clear = int(len(clear_valid) * VAL_SPLIT)

val_indices = cloudy_valid[:val_cloudy] + clear_valid[:val_clear]
train_indices = cloudy_valid[val_cloudy:] + clear_valid[val_clear:]

print(f"[✓] Train: {len(train_indices)} | Val: {len(val_indices)}")

# === FUNCTION TO PROCESS A SINGLE SAMPLE ===
def process_sample(idx, split):
    try:
        sample = dataset.read(idx)
        s2_l1c = sample.read(0)
        s2_label = sample.read(1)

        with rio.open(s2_l1c) as src:
            if src.count < 4:
                print(f"[{split}] Sample {idx} discarded: less than 4 bands.")
                return
            rgb = src.read([4, 3, 2])  # RGB

        with rio.open(s2_label) as lbl:
            mask = lbl.read(1)

        # === SAVE IMAGE ===
        img = np.transpose(rgb, (1, 2, 0))
        img = np.clip(img / 3000, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img_name = f"img_{idx:05d}.png"
        img.save(f"{SAVE_ROOT}/{split}/images/{img_name}")

        # === CREATE CLOUD MASK ===
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

    except Exception as e:
        print(f"[{split}] Error processing idx {idx}: {e}")

# === PROCESSING ===
print("🚀 Processing training set...")
for idx in tqdm(train_indices):
    process_sample(idx, "train")

print("🧪 Processing validation set...")
for idx in tqdm(val_indices):
    process_sample(idx, "val")
