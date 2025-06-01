from ultralytics import YOLO
import os

# === STEP 5: Validazione modelli ottenuti dal tuning ===
tune_dir = '../runs/tune/titan_hparam_tune'
weights_dir = os.path.join(tune_dir, 'weights')
results = []

print("\n📊 Validazione di tutti i modelli salvati durante il tuning...")

for file in os.listdir(weights_dir):
    if file.endswith('.pt'):
        model_path = os.path.join(weights_dir, file)
        model = YOLO(model_path)
        val_metrics = model.val(data='../yolo_configs/titan.yaml', split='val', verbose=False)

        results.append({
            'model': model_path,
            'metrics': val_metrics.box.map50  # puoi cambiare con .box.map, .seg.map, ecc.
        })

# Ordina per mAP@0.5 decrescente
results.sort(key=lambda x: x['metrics'], reverse=True)

best_model_path = results[0]['model']

print(f"✅ Miglior modello su validation: {best_model_path} (mAP@0.5: {results[0]['metrics']:.4f})")


# === STEP 6: Final Retraining on train + val (Optional but Recommended) ===
print("\n🔁 Final retraining on train + val (full_train)...")
final_model = YOLO(best_model_path)
final_model.train(
    data='../yolo_configs/titan_full.yaml',  # full_train split
    epochs=100,
    imgsz=416,
    batch = 16,
    project='../runs/train',
    name='titan_final_retrain'
)
# === STEP 7: Valutazione finale sul test set ===
#TODO for this part we must be consider if we have to use old test in replacement
print("\n🧪 Valutazione finale sul test set...")
final_model = YOLO(best_model_path)
metrics = final_model.val(data='../yolo_configs/titan.yaml', split='test')

print("🎯 Risultati sul test set:")
print(metrics)
