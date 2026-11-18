
from pathlib import Path
import cv2
from ultralytics import YOLO

# STEP 1: Define paths
project_dir = Path(__file__).resolve().parent.parent

video_path = project_dir / "data" / "videos" / "factory_video.mp4"

output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "tracked_factory_video.avi"

# STEP 2: Load YOLOv8
model = YOLO("yolov8n.pt")

# STEP 3: Open video
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

# STEP 5: Detect and track objects
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    resized_frame = cv2.resize(frame, (width, height))

    # YOLO + ByteTrack
    results = model.track(
    source=resized_frame,
    tracker=str(project_dir / "config" / "custom_bytetrack.yaml"),
    persist=True,
    conf=0.1,
    verbose=False
)

    # Draw bounding boxes and tracking IDs
    annotated_frame = results[0].plot()

    # Display frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    writer.write(annotated_frame)

    if frame_count % 30 == 0:
        print(f"Tracked {frame_count} frames")

# STEP 6: Release resources
cap.release()
writer.release()

print("Object tracking completed!")
print("Total frames:", frame_count)
print("Output saved to:", output_path)
