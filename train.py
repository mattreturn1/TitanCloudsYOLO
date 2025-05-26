from ultralytics import YOLO

"""
    | Parametro        | Descrizione                                      | Tipico intervallo |
    | ---------------- | ------------------------------------------------ | ----------------- |
    | `lr0`            | Learning rate iniziale                           | `(1e-5, 1e-2)`    | X
    | `lrf`            | Fattore di decay del learning rate               | `(0.01, 1.0)`     | X
    | `momentum`       | Momentum per SGD / Adam                          | `(0.6, 0.98)`     |
    | `weight_decay`   | Regularizzazione L2                              | `(0.0, 0.001)`    | X
    | `warmup_epochs`  | Epoche di warmup per il learning rate            | `(0, 5)`          |
    | `warmup_bias_lr` | Warmup learning rate per i bias                  | `(1e-6, 0.1)`     |
    | `box`            | Loss weight per bounding box                     | `(0.02, 0.2)`     |
    | `cls`            | Loss weight per classificazione                  | `(0.2, 4.0)`      |
    | `dfl`            | Distribution focal loss weight (solo per YOLOv8) | `(0.5, 3.0)`      |
    | `hsv_h`          | Augmentazione hue                                | `(0.0, 0.1)`      |
    | `hsv_s`          | Augmentazione saturazione                        | `(0.0, 0.9)`      |
    | `hsv_v`          | Augmentazione valore                             | `(0.0, 0.9)`      | A
    | `degrees`        | Augmentazione rotazione                          | `(0.0, 45.0)`     |
    | `translate`      | Augmentazione traslazione                        | `(0.0, 0.9)`      |
    | `scale`          | Augmentazione scala                              | `(0.0, 0.9)`      | A
    | `shear`          | Augmentazione shear                              | `(0.0, 10.0)`     |
    | `perspective`    | Augmentazione prospettiva                        | `(0.0, 0.001)`    |
    | `flipud`         | Flip verticale                                   | `(0.0, 1.0)`      | A
    | `fliplr`         | Flip orizzontale                                 | `(0.0, 1.0)`      | A
    | `mosaic`         | Probabilità di mosaic augment                    | `(0.0, 1.0)`      |
    | `mixup`          | Probabilità di mixup augment                     | `(0.0, 1.0)`      |
"""

def train_yolo():

    model = YOLO("yolo11n-seg.pt")

    search_space = {
        "lr0": (1e-5, 1e-3),
        "lrf": (0.01, 0.1),
        "weight_decay": (0.0001, 0.001)
    }

    model.tune(
        data="data.yaml",
        epochs=100,
        iterations=100,
        batch=8,
        optimizer="AdamW",
        space= search_space,
        plots=True,
        save=True,
        val=False,
        project="runs",
        name="cloud_yolo_seg_tuned",
        device=0,
    )

    model.train(
        data="data.yaml",
        imgsz=512,
        epochs=30,
        batch=32,
        name="cloud_yolo_seg",
        project="runs",
        device=0
    )

    print("Training completed.")

if __name__ == "__main__":
    train_yolo()
