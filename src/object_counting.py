
from pathlib import Path
import cv2

# Find the project folder
project_dir = Path(__file__).resolve().parent.parent

# Path to our sample image
image_path = project_dir / "data" / "images" / "sample.jpg"

# Read the image
image = cv2.imread(str(image_path))

# Check if the image loaded successfully
if image is None:
    print("Error: Image could not be loaded.")
else:
    print("Image loaded successfully!")

    # Get image dimensions
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Color channels:", channels)

    # Read the first pixel
    print("First pixel (BGR):", image[0, 0])
