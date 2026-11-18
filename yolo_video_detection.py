
from pathlib import Path
import cv2
from ultralytics import YOLO

# STEP 1: Define project paths
project_dir = Path(__file__).resolve().parent.parent

video_path = project_dir / "data" / "videos" / "factory_video.mp4"

output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "yolo_factory_detection.avi"

# STEP 2: Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# STEP 3: Open the factory video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError("Could not open factory video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS")

# STEP 4: Prepare output video
width = 640
height = 480

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video")

# STEP 5: Process each frame
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    resized_frame = cv2.resize(frame, (width, height))

    # Detect objects using YOLO
    results = model.predict(
        source=resized_frame,
        conf=0.25,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Add frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save processed frame to output video
    writer.write(annotated_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

# STEP 6: Release resources
cap.release()
writer.release()

print("YOLO video detection completed!")
print("Total frames processed:", frame_count)
print("Output saved to:", output_path)
