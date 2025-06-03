from ultralytics import YOLO  # Import the YOLO class from the Ultralytics library

# Define the hyperparameter search space for tuning
# Each key is a hyperparameter, and the value is a range (min, max) to explore
search_space = {
    "lr0": (1e-5, 1e-1),             # Initial learning rate
    "lrf": (1e-2, 1),                # Final learning rate multiplier
    "weight_decay": (0.0, 1e-3),     # Weight decay (L2 regularization)
    "hsv_v": (0.0, 0.9),             # Value (brightness) augmentation
    "translate": (0.0, 0.9),         # Image translation augmentation
    "scale": (0.0, 0.9),             # Image scaling augmentation
    "degrees":(0.0, 180),            # Image rotation augmentation
    "perspective":(0.0,0.01)         # Image perspective augmentation
}

# Load the best model trained on Earth clouds as the base for tuning
titan_model = YOLO('../runs/train/earth_clouds_yolo11n/weights/best.pt')

# Perform hyperparameter tuning on the Titan dataset
titan_model.tune(
    data='../yolo_configs/titan.yaml',  # Dataset configuration for Titan
    space=search_space,                 # Hyperparameter search space
    epochs=50,                          # Number of epochs for each trial
    imgsz=416,                          # Input image size
    batch=16,                           # Batch size
    iterations=80,                      # Number of tuning iterations (trials)
    optimizer='Adam',                   # Optimizer to use during tuning
    plots=True,                         # Enable result plots for visualization
    project='../runs/tune',             # Folder to save tuning results
    name='titan_hparam_tune'            # Name of the tuning experiment
)
