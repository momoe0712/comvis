import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model (update path to your custom model)
model = YOLO("--YOUR PRE-TRAINED PATH--")  # ex: gigi/runs/detect/train19/weights/best.pt

# Load the image (replace 'path_to_image.jpg' with the actual path to your image)
image_path = "--YOUR IMAGE PATH--" # ex: gigi/dataset/test/images/0071_jpg.rf.21ad37cd3c68265bb73add88bbb2cee9.jpg
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load the image.")
    exit()

# Define posture labels (assuming your model has specific class IDs for postures)
posture_labels = {0: 'Cavity', 1:'Filings', 2:'Impacted Tooth', 3: 'Implant'}

# Start measuring time
start_time = time.time()  # Record the start time

# Perform inference
results = model(image)

# End measuring time
end_time = time.time()  # Record the end time

# Calculate and print inference time
inference_time = end_time - start_time
print(f"Waktu inferensi: {inference_time:.4f} detik")

# Draw results on the image
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get the box coordinates
        x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID (assumed to be for posture)

        # Draw rectangle and label around the detected person with posture
        if cls in posture_labels:  # Ensure class ID corresponds to a posture
            posture = posture_labels[cls]  # Get the posture label
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'{posture}: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with the results
cv2.imshow('YOLOv8 Posture Detection', image)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()
