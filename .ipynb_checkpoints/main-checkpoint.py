from ultralytics import YOLO

# Load model
model = YOLO("/home/felixbaringin/gigi/pre-trained/yolo11n.pt")

# Custom training function with mAP50-95 early stopping
class EarlyStopOnMap:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def __call__(self, trainer):
        # Called every epoch
        metrics = trainer.metrics
        if metrics and 'metrics/mAP_50-95(B)' in metrics:
            current_map = metrics['metrics/mAP_50-95(B)']
            print(f"[Callback] Current mAP50-95: {current_map:.4f}")
            if current_map >= self.threshold:
                print(f"[Callback] mAP50-95 reached {current_map:.4f} (threshold: {self.threshold}) — stopping early.")
                raise StopIteration  # Gracefully stop training

# Tambahkan callback ini ke konfigurasi model
model.add_callback('on_train_epoch_end', EarlyStopOnMap(threshold=0.7))

# Start training
results = model.train(
    data="/home/felixbaringin/gigi/dataset/data.yaml",
    epochs=500,
    imgsz=640,
    device=0,
    workers=1  # Kurangi agar tidak melebihi shared memory
)
