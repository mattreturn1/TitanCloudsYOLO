from ultralytics import YOLO

# Define search space (corrected parameter names)
search_space = {
    "lr0": (1e-5, 1e-1),
    "lrf": (1e-2, 1),
    "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 1e-3),  # Underscore instead of space
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "translate": (0.0, 0.9),
    "scale": (0.0, 0.9),
    "flipud": (0.0, 1.0)  # Correct parameter name
}
# === STEP 4: Tuning iperparametri su Titano with data augmentation===
titan_model = YOLO('../runs/train/earth_clouds_yolo11n/weights/best.pt')
titan_model.tune(
    data='../yolo_configs/titan.yaml',
    space=search_space,  # Added space argument
    epochs=40,
    imgsz=416,
    batch = 16,
    iterations=40,
    optimizer='Adam',
    plots=True,
    project='../runs/tune',
    name='titan_hparam_tune'
)

