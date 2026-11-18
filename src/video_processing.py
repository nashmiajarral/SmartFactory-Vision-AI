
from pathlib import Path
import cv2

# STEP 1: Define project paths
project_dir = Path(__file__).resolve().parent.parent

video_path = project_dir / "data" / "videos" / "factory_video.mp4"
output_dir = project_dir / "outputs"

output_dir.mkdir(parents=True, exist_ok=True)

# STEP 2: Open the video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError(
        f"Cannot open video: {video_path}"
    )

# STEP 3: Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)
print("Video width:", width)
print("Video height:", height)
print("Total frames:", total_frames)

# STEP 4: Prepare output video
output_path = output_dir / "processed_video.avi"

fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (640, 480)
)

if fps <= 0 or not writer.isOpened():
    cap.release()
    writer.release()
    raise RuntimeError(
        "Invalid FPS or unable to create output video"
    )

# STEP 5: Process frames
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Resize the current frame
    resized_frame = cv2.resize(frame, (640, 480))

    # Add frame number
    cv2.putText(
        resized_frame,
        f"Frame: {frame_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save frame into output video
    writer.write(resized_frame)

# STEP 6: Release resources
cap.release()
writer.release()

print("Frames processed:", frame_count)
print("Output saved to:", output_path)
print("Video processing completed!")
