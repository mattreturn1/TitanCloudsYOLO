# Cloud Segmentation Project

## Step 1: Convert JSON Labels to TXT Format

Before training, you need to convert the polygon annotations from JSON (LabelMe format) to YOLO-compatible TXT format.

Run the conversion script:

```bash
python convert_labels.py
```

This will generate `.txt` label files corresponding to each image in your dataset.

## Step 2: Train the YOLO Model

Once the labels are converted, you can start training the model.

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=data.yaml epochs=100 imgsz=640
```

Make sure the `data.yaml` file is correctly set up to point to your training and validation datasets.

## Step 3: Test the YOLO Model

After training is complete, evaluate the model on the test set:

```bash
yolo task=segment mode=val model=runs/segment/train/weights/best.pt data=data.yaml
```

Replace the model path with the actual path to your trained weights.
