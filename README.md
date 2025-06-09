
# Cloud Segmentation with Titan and CloudSen12

This project focuses on the preparation and training of YOLO nano for cloud segmentation, using the **CloudSen12** and **Titan** datasets.

---

## Step 1: Dataset Preparation

### CloudSen12 Dataset Preprocessing

#### Description

- Filters samples with **cloud coverage > 55%**
- Uses only the **Sentinel-2 RGB bands**
- Extracts polygons of cloud classes **1 and 2** (these classes represent clouds)
- Converts masks into **YOLO-Segmentation** format

#### How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the script**:
   ```bash
   python preprocess_earth.py
   ```

3. **Output**:
   RGB images and `.txt` annotations in YOLO-Seg format saved in:
   ```
   datasets/CloudSen12/train/
   datasets/CloudSen12/val/
   ```

---

### Titan Dataset Preprocessing

#### Description

- Annotated using **LabelMe** (polygons)
- Automatically merges and splits into **train**, **val**, and **test**
- Converts `.json` annotations into **YOLO-Segmentation** format
- Removes corrupted images or those without valid labels
- Merges `train + val → full_train`

#### How to Run

1. Make sure the original dataset from https://zenodo.org/records/13988492 is placed inside the `datasets` directory, and each subdirectory `test` and `train`contain
only `labels` and `images`, others subdirectory must be deleted.

2. **Run the script**:
   ```bash
   python preprocess_titan.py
   ```

3. **Output**:
   Data and annotations in YOLO-Seg format saved in `datasets/Titan/`

---

## Step 2: Train and Tune the YOLO Model

### Installation

Install Ultralytics YOLOv11:
```bash
pip install ultralytics
```

and install all dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

### Training Workflow

Run the scripts in the following order:

1. `training_earth.py` – Initial training on CloudSen12
2. `tuning_titan.py` – Fine-tuning on the Titan dataset
3. `final_model.py` – Retrain the final model on the entire Titan dataset and evaluate its performance

Make sure the `.yaml` files in the `yolo_configs/` folder point to the correct dataset paths.

---

## Retuning or Retraining

To retrain or experiment with new parameters:

- Change the model names and weights in the scripts
- Update the `.yaml` files if needed

---

## Project Structure

```
.
├── datasets/
│   ├── CloudSen12/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   ├── Dataset_Zenodo/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │   │   ├── images/
│   │   │   └── labels/
│   └── Titan/
│       ├── full_train/
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/
│   ├── final_model.py
│   ├── preprocess_earth.py
│   ├── preprocess_titan.py
│   ├── training_earth.py
│   └── tuning_titan.py
├── yolo_configs/
│   ├── earth.yaml
│   ├── titan.yaml
│   └── titan_full.yaml
├── requirements.txt
└── README.md
```

---

## Notes
- To obtain a consistent subdataset from CloudSen12 in preprocessing phase we applied some controls in order to obtain valid images
which can make the download time-consuming (depending on your internet speed). We therefore recommend using the pre-uploaded version.
- Make sure to run the scripts from the correct directory
