import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import csv # Import the csv library

# --- CONFIGURATION ---
SOURCE_WEIGHTS_PATH = "best.pt"
SOURCE_VIDEO_PATH = "test_video.mp4"
TARGET_VIDEO_PATH = "tracking\\advanced_output.mp4"
TRACKER_CONFIG_PATH = "botsort_reid.yaml"
# NEW: Define the path for the trajectory data file
TRAJECTORY_CSV_PATH = "trajectories.csv"

def process_video():
    print("Starting video processing for data collection...")
    model = YOLO(SOURCE_WEIGHTS_PATH)
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    print("Initialized model and video info.")

    # NEW: A dictionary to store the history of each track
    track_history = defaultdict(list)

    results_generator = model.track(
        source=SOURCE_VIDEO_PATH,
        tracker=TRACKER_CONFIG_PATH,
        conf=0.1,
        iou=0.7,
        stream=True,
        verbose=False
    )

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame_idx, result in enumerate(results_generator):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)
            
            if detections.tracker_id is None:
                continue

            # NEW: Update track history with the center point of each bounding box
            for i in range(len(detections)):
                tracker_id = detections.tracker_id[i]
                bbox = detections.xyxy[i]
                center_x = (bbox + bbox[1]) / 2
                center_y = (bbox[2] + bbox[3]) / 2
                track_history[tracker_id].append((center_x, center_y))

            labels =
            for i in range(len(detections)):
                tracker_id = detections.tracker_id[i]
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]
                class_name = result.names[class_id]
                labels.append(f"{tracker_id} {class_name} {confidence:.2f}")
            
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            cv2.imshow("Advanced BotSort Tracking", annotated_frame)
            sink.write_frame(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # NEW: After processing, save the collected trajectories to a CSV file
    with open(TRAJECTORY_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'x', 'y'])
        for track_id, path in track_history.items():
            # Only save longer, more meaningful tracks
            if len(path) > 20: # Minimum path length to be considered for training
                for x, y in path:
                    writer.writerow([track_id, x, y])

    print(f"Video processing completed. Trajectory data saved to {TRAJECTORY_CSV_PATH}")
    cv2.destroyAllWindows()

process_video()
