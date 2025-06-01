from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library

# === STEP 3: Fine-tuning on Earth cloud images ===

# Load a pre-trained YOLO model with segmentation support
# '../yolo11n-seg.pt' is a custom or pre-trained model checkpoint
earth_model = YOLO('../yolo11n-seg.pt')

# Start training (fine-tuning) the model on a new dataset
earth_model.train(
    data='../yolo_configs/earth.yaml',     # Path to dataset configuration file (with classes, paths, etc.)
    epochs=100,                            # Train for 100 epochs
    imgsz=416,                             # Input image size (416x416)
    batch=16,                              # Batch size
    freeze=10,                             # Freeze the first 10 layers (useful for transfer learning)
    patience=10,                           # Stop training early if no improvement after 10 epochs
    project='../runs/train',               # Directory where training results will be saved
    name='earth_clouds_yolo11n',           # Name of the training run (used for output folders)
    pretrained=True                        # Use pre-trained weights as a starting point
)
