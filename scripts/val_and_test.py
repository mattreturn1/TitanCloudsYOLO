from ultralytics import YOLO  # Import YOLO class
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import cv2

# === STEP 5: Validate models obtained from hyperparameter tuning ===

tune_dir = '../runs/tune/titan_hparam_tune'  # Directory containing tuning results
weights_dir = os.path.join(tune_dir, 'weights')  # Path to saved model weights
results = []

print("\n📊 Validating all models saved during tuning...")

# Loop through all model weight files in the tuning directory
for file in os.listdir(weights_dir):
    if file.endswith('.pt'):
        model_path = os.path.join(weights_dir, file)
        model = YOLO(model_path)  # Load the model
        val_metrics = model.val(data='../yolo_configs/titan.yaml', split='val', verbose=False)  # Evaluate on val set

        results.append({
            'model': model_path,
            'metrics': val_metrics.box.map50  # Use mAP@0.5; change to .box.map or .seg.map if needed
        })

# Sort results by mAP@0.5 in descending order
results.sort(key=lambda x: x['metrics'], reverse=True)
best_model_path = results[0]['model']

print(f"✅ Best model on validation set: {best_model_path} (mAP@0.5: {results[0]['metrics']:.4f})")

# === STEP 6: Final retraining on train + val (Optional but Recommended) ===

print("\n🔁 Final retraining on train + val (full_train)...")
final_model = YOLO(best_model_path)  # Load best model
final_model.train(
    data='../yolo_configs/titan_full.yaml',  # Train + validation split configuration
    epochs=500,
    imgsz=416,
    batch=16,
    project='../runs/train',
    name='titan_final_retrain'
)

# === STEP 7: Final evaluation on the custom test set ===

# Function to compute Intersection over Union (IoU)
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

# Custom evaluation function
def run_inference_and_metrics(model_path):
    print("\n🧪 Final evaluation on the custom test set...")
    model = YOLO(model_path)  # Load model for testing

    # Define test and output directories
    test_images_dir = "../datasets/Titan/test/images"
    label_dir = "../datasets/Titan/test/labels"
    output_dir = "../predictions"
    os.makedirs(output_dir, exist_ok=True)

    # Lists and counters to store evaluation metrics
    ious, precs, recalls, accs = [], [], [], []
    global_tp, global_fp, global_fn = 0, 0, 0
    tot = 0
    processed = 0

    # Loop over each test image
    for file in os.listdir(test_images_dir):
        if not file.endswith(".png"):
            continue

        image_path = os.path.join(test_images_dir, file)
        label_path = os.path.join(label_dir, file.replace(".png", ".txt"))
        if not os.path.exists(label_path):
            print(f"Label file missing for {file}, skipping")
            continue

        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Run inference
        results = model.predict(source=image_path, save=False, conf=0.5)
        result = results[0]

        # Initialize predicted mask
        pred_mask = np.zeros((height, width), dtype=bool)
        if result.masks is not None:
            for m in result.masks.data:
                mask = m.cpu().numpy().astype(bool)
                if mask.shape != pred_mask.shape:
                    # Resize if shape mismatch
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                pred_mask = np.logical_or(pred_mask, mask)

        # Load ground truth mask from polygon annotations
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
                cv2.fillPoly(gt_mask, [pts.reshape(-1, 1, 2)], 1)

        # Convert ground truth to boolean
        gt_mask_bool = gt_mask.astype(bool)

        # Flatten masks for metric computation
        y_true = gt_mask_bool.flatten()
        y_pred = pred_mask.flatten()

        # Compute metrics for the current image
        ious.append(compute_iou(gt_mask_bool, pred_mask))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true, y_pred))

        # Update global counters
        tp = np.logical_and(y_true, y_pred).sum()
        fp = np.logical_and(~y_true, y_pred).sum()
        fn = np.logical_and(y_true, ~y_pred).sum()
        global_tp += tp
        global_fp += fp
        global_fn += fn
        tot += len(y_true)
        processed += 1

        # Save binary prediction mask
        pred_mask_uint8 = (pred_mask.astype(np.uint8)) * 255
        cv2.imwrite(os.path.join(output_dir, file.replace(".png", "_pred_mask.png")), pred_mask_uint8)

        # Create color overlay visualization
        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = pred_mask_uint8  # Green channel
        overlay = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)

        # Draw predicted bounding boxes (if available)
        if result.boxes is not None:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save overlay image
        cv2.imwrite(os.path.join(output_dir, file.replace(".png", "_overlay.png")), overlay)

    # Print evaluation summary
    print(f"\nProcessed {processed} images.")
    print("Metrics:")
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
        tn = tot - global_tp - global_fp - global_fn
        global_acc = (global_tp + tn) / tot
        print(f"global_accuracy: {global_acc:.4f}")
    else:
        print("global_accuracy: N/A")

# === Run the final evaluation ===
run_inference_and_metrics(best_model_path)
