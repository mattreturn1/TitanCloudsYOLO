# Cloud Segmentation Project

## Step 1:Dataset Preparation



### 🌍 CloudSen12 Dataset Preprocessing

#### 📋 Description

- Filters only samples with cloud coverage > 65%.
- Uses Sentinel-2 RGB bands.
- Extracts polygons of cloud classes 1 and 2.
- Converts masks into YOLO Segmentation format.

#### ▶️ How to Run

1. Install required libraries:
   ```bash
   pip install tacoreader rasterio tqdm pillow opencv-python
    ```
2. Execute preprocess_earth.py
3. Output:
YOLO-segmentation formatted labels and RGB images saved into `datasets/CloudSen12/train` and `datasets/CloudSen12/val`.

### 🪐 Titan Dataset Preprocessing

#### 📋 Description

- Original dataset annotated using LabelMe.
- Dataset is first merged, split into train/val/test.
- Converts polygon annotations from .json to YOLO-seg format.
- Removes corrupted images and entries without valid labels.
- Optionally merges train and val into full_train for complete training set.

#### ▶️ How to Run

1. Ensure dataset subrepositories `train` and `test` contain only images and labels repo:
2. Execute `preprocess_titan.py`
3. Output: YOLO-segmentation formatted labels and images saved into `datasets/Titan`

## Step 2: Train and tune the YOLO Model
Once the labels are converted, you can start training the model.

1. Install Ultralytics YOLO:
```bash
pip install ultralytics
```
2. Execute `preprocess_earth.py`

Make sure the `.yaml` files in `yolo_configs` repo are correctly set up to point to your training and validation datasets.
