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
python train_yolo.py
```

Make sure the `data.yaml` file is correctly set up to point to your training and validation datasets.

## Step 3: Test the YOLO Model

After training is complete, evaluate the model on the test set:

```bash
python test.py
```

Replace the model path with the actual path to your trained weights.
