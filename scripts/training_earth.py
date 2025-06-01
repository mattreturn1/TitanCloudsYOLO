from ultralytics import YOLO
# === STEP 3: Fine-tuning su nuvole terrestri ===
earth_model = YOLO('../yolo11n-seg.pt')

earth_model.train(
    data='../yolo_configs/earth.yaml',
    epochs=100,
    imgsz=416,
    batch = 16,
    freeze=10,
    patience=10,
    project='../runs/train',
    name='earth_clouds_yolo11n',
    pretrained=True
)