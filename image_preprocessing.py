
from pathlib import Path
import cv2

# STEP 1: Define project paths
project_dir = Path(__file__).resolve().parent.parent

image_path = project_dir / "data" / "images" / "sample.jpg"
output_dir = project_dir / "outputs"

# Create outputs folder if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# STEP 2: Load the original image
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

print("Original image shape:", image.shape)

# STEP 3: Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Grayscale image shape:", gray_image.shape)

# STEP 4: Resize the original image
resized_image = cv2.resize(image, (640, 480))

print("Resized image shape:", resized_image.shape)

# STEP 5: Save processed images
gray_path = output_dir / "grayscale.jpg"
resized_path = output_dir / "resized.jpg"

gray_saved = cv2.imwrite(str(gray_path), gray_image)
resized_saved = cv2.imwrite(str(resized_path), resized_image)

if not gray_saved or not resized_saved:
    raise RuntimeError("Failed to save processed images")

print("Grayscale image saved successfully!")
print("Resized image saved successfully!")
print("Image preprocessing completed!")
