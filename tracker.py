# custom_tracker.py

import supervision as sv
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION ---
# Set the paths for your custom model, source video, and desired output video
MODEL_PATH = "path/to/your/custom_model.pt"  # <-- IMPORTANT: Update this path
SOURCE_VIDEO_PATH = "path/to/your/input_video.mp4" # <-- Update this path
TARGET_VIDEO_PATH = "path/to/your/output_video.mp4" # <-- Update this path

# --- INITIALIZATION ---
print("Loading your custom model...")
# Load your custom-trained YOLOv8 model
model = YOLO(MODEL_PATH)

# Initialize the ByteTrack algorithm provided by the supervision library [4]
# The parameters can be fine-tuned for performance. For example:
# track_thresh: The confidence threshold for detections to be considered high-confidence.
# track_buffer: The number of frames to keep a track alive without a new detection.
# match_thresh: The IOU threshold for matching detections to existing tracks.
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8)

# Initialize supervision's annotators for drawing boxes and labels
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

print("Model and tracker have been initialized successfully.")

# --- VIDEO PROCESSING ---
# This function will be called for each frame in the video
def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
    """
    Processes a single video frame to detect and track objects.

    Args:
        frame (np.ndarray): The video frame to process.
        frame_index (int): The index of the current frame.

    Returns:
        np.ndarray: The annotated frame with tracking information.
    """
    # 1. DETECTION: Run your custom model on the frame
    results = model(frame) # Get results for the first image

    # 2. DATA CONVERSION: Convert detection results to a standard format
    # The supervision library has a convenient converter for ultralytics results [4]
    detections = sv.Detections.from_ultralytics(results)

    # 3. TRACKING: Update the tracker with the new detections
    # This is the core step where ByteTrack applies its logic to associate detections
    # with existing tracks, handling occlusions and preventing ID switches.[1, 3]
    tracked_detections = byte_tracker.update_with_detections(detections)

    # 4. ANNOTATION: Prepare labels and draw on the frame
    # Create descriptive labels for each tracked object, including its unique ID
    labels =} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in tracked_detections
    ]

    # Annotate the frame with bounding boxes for the tracked objects
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=tracked_detections
    )

    # Annotate the frame with the generated labels
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=tracked_detections,
        labels=labels
    )

    return annotated_frame

# --- MAIN EXECUTION SCRIPT ---
if __name__ == "__main__":
    print(f"Starting tracking on video: {SOURCE_VIDEO_PATH}")

    # Use supervision's process_video utility to handle the video processing loop [4]
    # It reads the source video, applies the 'process_frame' callback to each frame,
    # and writes the annotated frames to the target video path.
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=process_frame
    )

    print(f"Tracking complete. The output video is saved at: {TARGET_VIDEO_PATH}")
