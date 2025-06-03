from ultralytics import YOLO
import os
import shutil

base_tune_dir = '../runs/tune'  # Directory containing subfolders like train1, train2, etc.
results = []

print("\n🔍 Validating all best.pt models saved during tuning...")

# Iterate over subdirectories
for subdir in os.listdir(base_tune_dir):
    subdir_path = os.path.join(base_tune_dir, subdir)
    weights_path = os.path.join(subdir_path, 'weights', 'best.pt')

    if os.path.isfile(weights_path):
        print(f"Validating: {weights_path}")
        model = YOLO(weights_path)
        val_metrics = model.val(data='../yolo_configs/titan.yaml', split='val', verbose=False)

        # Clean up runs directory after each validation
        if os.path.exists("runs"):
            shutil.rmtree("runs")

        results.append({
            'model': weights_path,
            'metrics': val_metrics.box.map50  # mAP@0.5
        })

# Sort and select best
results.sort(key=lambda x: x['metrics'], reverse=True)
best_model_path = results[0]['model']

print(f"\nBest model on validation set:\n{best_model_path} (mAP@0.5: {results[0]['metrics']:.4f})")

# Save path to file
with open("best_model.txt", "w") as f:
    f.write(best_model_path)
