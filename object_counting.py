
from pathlib import Path
import cv2
from ultralytics import YOLO

# STEP 1: Define project paths
project_dir = Path(__file__).resolve().parent.parent

video_path = project_dir / "data" / "videos" / "factory_video.mp4"
tracker_path = project_dir / "config" / "custom_bytetrack.yaml"

output_dir = project_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "counted_factory_fixed.avi"

# STEP 2: Load YOLO model
model = YOLO("yolov8n.pt")

# STEP 3: Open input video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid video FPS")

# STEP 4: Prepare output video
width, height = 640, 480

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

# STEP 5: Initialize counting variables
line_x = width // 2

previous_positions = {}
counted_ids = set()

total_count = 0
frame_count = 0

print("Starting object counting...")
print("Counting direction: RIGHT to LEFT")
print("Counting line X:", line_x)

# STEP 6: Process video frames
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    resized_frame = cv2.resize(frame, (width, height))

    # YOLO detection + ByteTrack tracking
    results = model.track(
        source=resized_frame,
        tracker=str(tracker_path),
        persist=True,
        conf=0.05,
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()

    # STEP 7: Check tracked objects
    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box

            # Calculate object center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check previous position
            if track_id in previous_positions:

                previous_x = previous_positions[track_id]

                # Count RIGHT-to-LEFT crossing
                if (
                    previous_x > line_x
                    and center_x <= line_x
                    and track_id not in counted_ids
                ):
                    counted_ids.add(track_id)
                    total_count += 1

                    print(
                        f"Frame {frame_count}: "
                        f"Object ID {track_id} crossed "
                        f"from X={previous_x} to X={center_x}. "
                        f"Total Count: {total_count}"
                    )

                    # Mark the counted object
                    cv2.circle(
                        annotated_frame,
                        (center_x, center_y),
                        8,
                        (0, 255, 255),
                        -1
                    )

            # Update position for next frame
            previous_positions[track_id] = center_x

    # STEP 8: Draw red counting line
    cv2.line(
        annotated_frame,
        (line_x, 0),
        (line_x, height - 1),
        (0, 0, 255),
        5
    )

    # Display counting line label
    cv2.putText(
        annotated_frame,
        "COUNTING LINE",
        (line_x + 10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # Display total count
    cv2.putText(
        annotated_frame,
        f"Total Count: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display frame number
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Write processed frame
    writer.write(annotated_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames")

# STEP 9: Release resources
cap.release()
writer.release()

print("\nCounting completed!")
print("Total frames:", frame_count)
print("Final object count:", total_count)
print("Output saved to:", output_path)
