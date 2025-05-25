from ultralytics import YOLO
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, accuracy_score
import cv2

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def run_inference_and_metrics():
    model = YOLO("runs/cloud_yolo_seg3/weights/best.pt")

    test_images_dir = "Dataset1/test/images"
    label_dir = "Dataset1/test/labels"
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    ious, precs, recalls, accs = [], [], [], []
    global_tp, global_fp, global_fn = 0, 0, 0
    tot = 0

    for file in os.listdir(test_images_dir):
        if not file.endswith(".png"):
            continue

        image_path = os.path.join(test_images_dir, file)
        label_path = os.path.join(label_dir, file.replace(".png", ".txt"))
        if not os.path.exists(label_path):
            print(f"Label file missing for {file}, skipping")
            continue

        # Carica immagine e dimensioni
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Predict
        results = model.predict(source=image_path, save=False, conf=0.5)

        # Costruisci maschera predetta
        pred_mask = np.zeros((height, width), dtype=bool)
        for r in results:
            if r.masks is not None:
                for m in r.masks.data:
                    mask = m.cpu().numpy().astype(bool)
                    # ATTENZIONE: la dimensione della maschera deve essere width x height
                    # Se maschera ha dimensione diversa, va ridimensionata o gestita di conseguenza
                    if mask.shape != pred_mask.shape:
                        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                    pred_mask = np.logical_or(pred_mask, mask)

        # Costruisci maschera ground truth
        gt_mask = np.zeros((height, width), dtype=np.uint8)
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                points = list(map(float, parts[1:]))
                pts = np.array(points).reshape(-1, 2)
                pts[:, 0] *= width
                pts[:, 1] *= height
                pts = pts.astype(np.int32)
                cv2.fillPoly(gt_mask, [pts], 1)

        gt_mask_bool = gt_mask.astype(bool)

        # Metriche per immagine
        iou = compute_iou(gt_mask_bool, pred_mask)
        ious.append(iou)

        y_true = gt_mask_bool.flatten()
        y_pred = pred_mask.flatten()

        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))

        tp = np.logical_and(y_true == True, y_pred == True).sum()
        fp = np.logical_and(y_true == False, y_pred == True).sum()
        fn = np.logical_and(y_true == True, y_pred == False).sum()
        global_tp += tp
        global_fp += fp
        global_fn += fn
        tot += len(y_true)

        # Salva maschera predetta (0-255)
        pred_mask_uint8 = (pred_mask.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(output_dir, file.replace(".png", "_pred_mask.png")), pred_mask_uint8)

        # Salva overlay maschera su immagine originale
        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = pred_mask_uint8  # verde
        overlay = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)

        # Aggiungi bounding boxes (se presenti)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes.xyxy.cpu().numpy():  # xyxy: xmin, ymin, xmax, ymax
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rosso

        cv2.imwrite(os.path.join(output_dir, file.replace(".png", "_overlay.png")), overlay)

    print("Metriche:")
    print(f"avg_iou: {np.mean(ious):.4f}")
    print(f"avg_precision: {np.mean(precs):.4f}")
    print(f"avg_recall: {np.mean(recalls):.4f}")
    print(f"avg_accuracy: {np.mean(accs):.4f}")
    if global_tp + global_fp > 0:
        print(f"global_precision: {global_tp / (global_tp + global_fp):.4f}")
    else:
        print("global_precision: N/A")
    if global_tp + global_fn > 0:
        print(f"global_recall: {global_tp / (global_tp + global_fn):.4f}")
    else:
        print("global_recall: N/A")
    if tot > 0:
        global_acc = (global_tp + (tot - global_tp - global_fp - global_fn)) / tot
        print(f"global_accuracy: {global_acc:.4f}")
    else:
        print("global_accuracy: N/A")

if __name__ == "__main__":
    run_inference_and_metrics()
