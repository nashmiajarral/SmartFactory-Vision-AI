# SmartFactory Vision AI — Industrial Object Detection, Tracking & Counting

A computer vision project developed using **Python, OpenCV, YOLOv8, and ByteTrack** to detect, track, and count moving objects in industrial conveyor-belt video footage.

This project explores how computer vision can support factory automation by monitoring objects, maintaining tracking IDs, and counting packages as they cross a virtual line.

## Project Overview

SmartFactory Vision AI processes recorded conveyor-belt footage and applies object detection and multi-object tracking to identify moving objects.

The system uses YOLOv8 for object detection, ByteTrack for tracking, and OpenCV for video processing, visualization, and automated line-crossing counting.

The project was developed as a hands-on introduction to computer vision and industrial AI applications.

## Key Features

- **Image Processing:** Image loading, resizing, grayscale conversion, and preprocessing using OpenCV.
- **Video Processing:** Frame-by-frame video processing and annotated video generation.
- **Object Detection:** Detection of objects in images and videos using a pretrained YOLOv8 model.
- **Object Tracking:** Multi-object tracking using ByteTrack with unique tracking IDs.
- **Custom Tracker Configuration:** Adjusted tracking parameters to improve ID persistence.
- **Object Counting:** Automated counting when tracked objects cross a virtual line from right to left.
- **Visual Output:** Bounding boxes, tracking IDs, counting lines, and running object counts displayed on processed video.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image and video processing |
| YOLOv8 | Object detection |
| Ultralytics | YOLO model implementation |
| ByteTrack | Multi-object tracking |
| PyTorch | Deep learning framework used by YOLO |
| NumPy | Numerical operations |

## Project Structure

```text
SmartFactory-Vision-AI/
│
├── config/
│   └── custom_bytetrack.yaml
│
├── src/
│   ├── image_basics.py
│   ├── image_preprocessing.py
│   ├── video_processing.py
│   ├── yolo_detection.py
│   ├── yolo_video_detection.py
│   ├── object_tracking.py
│   └── object_counting.py
│
├── data/
│   ├── images/
│   └── videos/
│       └── factory_video.mp4
│
├── outputs/
│
├── requirements.txt
└── README.md
```

**Note:** Input datasets and generated output files may be excluded from the repository due to file size.

## How the System Works

**1. Video Input**

The system reads conveyor-belt video footage using OpenCV.

**2. Object Detection**

YOLOv8 analyzes each frame and predicts bounding boxes around recognized objects.

**3. Object Tracking**

ByteTrack associates detections across consecutive frames and assigns tracking IDs.

**4. Virtual Counting Line**

A vertical counting line is placed in the video frame.

**5. Object Counting**

When a tracked object's center moves across the line from right to left, the system increases the count. Previously counted tracking IDs are stored to avoid counting the same ID multiple times.

**6. Output Generation**

The processed video is saved with detection boxes, tracking IDs, a counting line, and a running object count.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartFactory-Vision-AI.git
cd SmartFactory-Vision-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Input Video

Place a conveyor-belt video inside:

```text
data/videos/factory_video.mp4
```

Create the directories if they do not already exist.

### 5. Run Object Detection

```bash
python src/yolo_video_detection.py
```

### 6. Run Object Tracking

```bash
python src/object_tracking.py
```

### 7. Run Object Counting

```bash
python src/object_counting.py
```

The system processes the input video and saves annotated results in the `outputs/` directory.

## Experimental Results

The object-counting pipeline was tested on a conveyor-belt video containing **607 frames**.

The system successfully demonstrated:

- Object detection in recorded factory footage.
- Tracking ID assignment and persistence.
- Virtual line-crossing detection.
- Automated counting of tracked objects.
- Annotated video generation.

During experimentation, changes to the tracking configuration improved the detected crossing count from **2 to 4**.

This count represents detected crossing events and is not a verified total of all physical packages in the video.

## Known Limitations

- The pretrained YOLOv8 model is not specifically trained to recognize generic cardboard packages.
- Some packages are detected inconsistently.
- Tracking IDs may temporarily disappear when detections are lost.
- Objects may cross the counting line while undetected, resulting in missed counts.
- Counting accuracy depends on object detection quality, tracking stability, and counting-line placement.

## Future Improvements

- Train a custom YOLO model using annotated cardboard-package images.
- Improve detection performance for industrial objects.
- Explore alternative tracking and counting approaches.
- Evaluate counting accuracy against manually labeled ground-truth data.
- Extend the system to support real-time camera feeds.

## Learning Outcomes

Through this project, I gained practical experience in:

- Computer vision fundamentals.
- Image and video processing using OpenCV.
- Object detection using YOLOv8.
- Multi-object tracking using ByteTrack.
- Implementing line-crossing counting logic.
- Configuring and testing tracking parameters.
- Debugging detection and tracking limitations.

## Applications

The techniques explored in this project are relevant to:

- Manufacturing automation.
- Conveyor-belt monitoring.
- Warehouse object tracking.
- Industrial product counting.
- Computer vision-based monitoring systems.

## Project Status

**Completed — Beginner Computer Vision Project**

The core detection, tracking, and counting pipeline is implemented. Further improvements to detection reliability and counting accuracy are identified as future work.

## Author

Software Engineering Graduate

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and practical AI applications.
