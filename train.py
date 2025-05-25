from ultralytics import YOLO

#AUGMENTATION ?
def train_yolo():
    model = YOLO("yolo11n-seg.pt")

    """
    | Parametro        | Descrizione                                      | Tipico intervallo |
    | ---------------- | ------------------------------------------------ | ----------------- |
    x| `lr0`            | Learning rate iniziale                           | `(1e-5, 1e-2)`    |
    x| `lrf`            | Fattore di decay del learning rate               | `(0.01, 1.0)`     |
    | `momentum`       | Momentum per SGD / Adam                          | `(0.6, 0.98)`     |
    x| `weight_decay`   | Regularizzazione L2                              | `(0.0, 0.001)`    |
    | `warmup_epochs`  | Epoche di warmup per il learning rate            | `(0, 5)`          |
    | `warmup_bias_lr` | Warmup learning rate per i bias                  | `(1e-6, 0.1)`     |
    | `box`            | Loss weight per bounding box                     | `(0.02, 0.2)`     |
    | `cls`            | Loss weight per classificazione                  | `(0.2, 4.0)`      |
    | `dfl`            | Distribution focal loss weight (solo per YOLOv8) | `(0.5, 3.0)`      |
    | `hsv_h`          | Augmentazione hue                                | `(0.0, 0.1)`      |
    | `hsv_s`          | Augmentazione saturazione                        | `(0.0, 0.9)`      |
    v| `hsv_v`          | Augmentazione valore                             | `(0.0, 0.9)`      |
    | `degrees`        | Augmentazione rotazione                          | `(0.0, 45.0)`     |
    | `translate`      | Augmentazione traslazione                        | `(0.0, 0.9)`      |
    v| `scale`          | Augmentazione scala                              | `(0.0, 0.9)`      |
    | `shear`          | Augmentazione shear                              | `(0.0, 10.0)`     |
    | `perspective`    | Augmentazione prospettiva                        | `(0.0, 0.001)`    |
    v| `flipud`         | Flip verticale                                   | `(0.0, 1.0)`      |
    v| `fliplr`         | Flip orizzontale                                 | `(0.0, 1.0)`      |
    | `mosaic`         | Probabilità di mosaic augment                    | `(0.0, 1.0)`      |
    | `mixup`          | Probabilità di mixup augment                     | `(0.0, 1.0)`      |
    """
    search_space = {
        "lr0": (1e-5, 1e-3),
        "lrf": (0.01, 0.1),
        "momentum": (0.7, 0.98),
        "weight_decay": (0.0001, 0.001),
        "warmup_epochs": (0, 3),
        "box": (0.05, 0.2),
        "cls": (0.2, 2.0),
        "dfl": (0.5, 1.5),
        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.5),
        "hsv_v": (0.0, 0.5),
        "degrees": (0.0, 20.0),
        "translate": (0.0, 0.2),
        "scale": (0.5, 1.5),
        "fliplr": (0.0, 1.0),
        "mosaic": (0.5, 1.0),
        "mixup": (0.0, 0.5),
    }

    # Ottimizza con batch ridotto
    model.tune(
        data="cloud.yaml",
        epochs=100,
        iterations=10,
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
        data="cloud.yaml",
        imgsz=512,
        epochs=30,
        batch=32,
        name="cloud_yolo_seg",
        project="runs",
        device=0
    )

    print("Training completato.")

if __name__ == "__main__":
    train_yolo()
