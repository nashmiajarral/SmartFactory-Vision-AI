from pathlib import Path
import cv2
from ultralytics import YOLO

# Step 1: Define project paths
project_dir = Path(__file__).resolve().parent.parent
image_path = project_dir / "data" / "images" / "sample.jpg"
output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Step 3: Run object detection
results = model.predict(source=str(image_path), conf=0.25)

# Step 4: Draw bounding boxes and labels
annotated_image = results[0].plot()

# Step 5: Save the detection result
output_path = output_dir / "yolo_detection.jpg"

if not cv2.imwrite(str(output_path), annotated_image):
    raise RuntimeError("Could not save detection image")

print("Detection completed!")
print("Output saved to:", output_path)
