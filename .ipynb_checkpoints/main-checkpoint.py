from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("home/felixbaringin/gigi/pre-trained/yolo11x.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/felixbaringin/gigi/dataset/data.yaml", epochs=200, imgsz=640, patience=7)

