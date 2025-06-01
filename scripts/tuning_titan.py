from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library

# Define the hyperparameter search space for tuning
# Each key is a hyperparameter, and the value is a range (min, max) to explore
search_space = {
    "lr0": (1e-5, 1e-1),             # Initial learning rate
    "lrf": (1e-2, 1),                # Final learning rate multiplier
    "momentum": (0.6, 0.98),         # Momentum for SGD optimizer
    "weight_decay": (0.0, 1e-3),     # Weight decay (L2 regularization)
    "hsv_h": (0.0, 0.1),             # Hue augmentation
    "hsv_s": (0.0, 0.9),             # Saturation augmentation
    "hsv_v": (0.0, 0.9),             # Value (brightness) augmentation
    "translate": (0.0, 0.9),         # Image translation augmentation
    "scale": (0.0, 0.9),             # Image scaling augmentation
    "flipud": (0.0, 1.0)             # Vertical flip probability
}

# === STEP 4: Hyperparameter tuning on Titan dataset with data augmentation ===

# Load the best model trained on Earth clouds as the base for tuning
titan_model = YOLO('../runs/train/earth_clouds_yolo11n/weights/best.pt')

# Perform hyperparameter tuning on the Titan dataset
titan_model.tune(
    data='../yolo_configs/titan.yaml',  # Dataset configuration for Titan
    space=search_space,                 # Hyperparameter search space
    epochs=40,                          # Number of epochs for each trial
    imgsz=416,                          # Input image size
    batch=16,                           # Batch size
    iterations=40,                      # Number of tuning iterations (trials)
    optimizer='Adam',                   # Optimizer to use during tuning
    plots=True,                         # Enable result plots for visualization
    project='../runs/tune',             # Folder to save tuning results
    name='titan_hparam_tune'            # Name of the tuning experiment
)
