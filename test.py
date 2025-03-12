from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("D://post_kuliah/test/yolov8n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="D://post_kuliah/test/dataset/data.yaml", epochs=10, imgsz=640)