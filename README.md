YOLO-Based Multi-Object Tracking & Fine-Tuning

This repository provides a streamlined pipeline for training a state-of-the-art YOLO model and using it to perform real-time multi-object tracking (MOT) on video streams using ByteTrack.

🚀 Features
Custom Training: Fine-tune a YOLO object detection model with layer freezing capabilities.

ByteTrack Integration: High-performance, multi-object tracking using detection confidence scores.

Video Reconstruction: Automatically processes input video frames and compiles them back into a tracked .mp4 output file.

🛠️ Prerequisites & Installation
Ensure you have Python 3.8+ installed along with the required deep learning and computer vision libraries.

Bash
pip install ultralytics opencv-python torch
📦 Project Setup
Your dataset configuration should follow the standard Ultralytics YOLO format. Ensure your custom your_dataset.yaml is pointing to the correct paths:

YAML
# Example: your_dataset.yaml
path: ../datasets/my_data
train: images/train
val: images/val

names:
  0: object_class_A
  1: object_class_B
💻 Usage
1. Model Training & Fine-Tuning
To train or fine-tune the model on your custom dataset while freezing the backbone layers, use the following snippet:

Python
from ultralytics import YOLO

# Load the base model
model = YOLO("yolo26n.pt")

# Fine-tune the model
model.train(
    data="your_dataset.yaml",
    epochs=50,
    imgsz=640,
    freeze=22,  # Freezes the backbone layers to preserve features
    batch=16,
    device=0    # Uses GPU 0; change to 'cpu' if no GPU is available
)
2. Multi-Object Tracking Pipeline
Pass a video path to the tracking function to extract frames, track objects frame-by-frame, and export the annotated video.

Python
import cv2
from ultralytics import YOLO

def multi_object_tracking(video_path):
    # Load the fine-tuned model
    model = YOLO("yolo26n.pt")
    output_path = 'output.mp4'
    
    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(video_path)

    # Check for video load error
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open the video file: {video_path}")

    # Extract video properties for reconstruction
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video frames with ByteTrack... please wait.")

    # Processing loop
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Persistent tracking using ByteTrack
            result = model.track(frame, tracker="bytetrack.yaml", persist=True)

            # Draw tracking bounding boxes and IDs
            ready_frame = result[0].plot()

            # Write the annotated frame to output video
            out.write(ready_frame)
        else:
            break

    # Release resources
    out.release()
    cap.release()
    
    print(f"Tracking complete. Output saved to {output_path}")
    return output_path

# Execute tracking
multi_object_tracking("input_video.mp4")
⚙️ How It Works
Layer Freezing: Setting freeze=22 keeps the initial weights of the pretrained network intact. This speeds up training and prevents overfitting if you are working with a smaller dataset.

ByteTrack Tracker: The bytetrack.yaml tracker assigns unique IDs to detected objects and maintains their identities across frames, even through minor occlusions.
