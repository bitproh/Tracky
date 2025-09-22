import cv2
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# --- CONFIGURATION ---
SOURCE_WEIGHTS_PATH = "best.pt"
SOURCE_VIDEO_PATH = "test_video.mp4"
TARGET_VIDEO_PATH = "tracking\\advanced_output_with_ghosts.mp4"
TRACKER_CONFIG_PATH = "botsort_reid.yaml"
LSTM_MODEL_PATH = "lstm_predictor.pth"
INPUT_SEQUENCE_LENGTH = 20
OUTPUT_SEQUENCE_LENGTH = 50

# --- LSTM MODEL DEFINITION (must match the training script) ---
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2):
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, future=OUTPUT_SEQUENCE_LENGTH):
        outputs =
        _, (h_n, c_n) = self.lstm(x)
        decoder_input = x[:, -1, :]
        for _ in range(future):
            output, (h_n, c_n) = self.lstm(decoder_input.unsqueeze(1), (h_n, c_n))
            output = self.linear(output.squeeze(1))
            outputs.append(output)
            decoder_input = output
        return torch.stack(outputs, 1)

def process_video_with_ghost_tracking():
    print("Starting video processing with Ghost Tracking...")
    model = YOLO(SOURCE_WEIGHTS_PATH)
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # --- LOAD AND PREPARE LSTM ---
    lstm_model = TrajectoryLSTM()
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    lstm_model.eval()
    
    # Prepare a scaler based on some sample data to inverse transform predictions
    # In a real application, you would save and load the scaler used during training
    scaler = MinMaxScaler()
    # Fit with dummy data that represents the pixel coordinate space
    scaler.fit(np.array([, [video_info.width, video_info.height]]))

    # --- TRACKING DATA STRUCTURES ---
    track_history = defaultdict(list)
    lost_tracks = {} # {tracker_id: { "history": [...], "frames_lost": 0 }}
    
    results_generator = model.track(
        source=SOURCE_VIDEO_PATH,
        tracker=TRACKER_CONFIG_PATH,
        conf=0.1, iou=0.7, stream=True, verbose=False
    )

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for result in results_generator:
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)
            
            current_track_ids = set()
            if detections.tracker_id is not None:
                current_track_ids = set(detections.tracker_id)
                for i in range(len(detections)):
                    tracker_id = detections.tracker_id[i]
                    bbox = detections.xyxy[i]
                    center_x = (bbox + bbox[1]) / 2
                    center_y = (bbox[2] + bbox[3]) / 2
                    track_history[tracker_id].append((center_x, center_y))
                    
                    # If this track was previously lost, remove it from the lost list
                    if tracker_id in lost_tracks:
                        del lost_tracks[tracker_id]

            # --- GHOST TRACKING LOGIC ---
            # Identify tracks that were present in the last frame but not this one
            previous_track_ids = set(track_history.keys())
            newly_lost_ids = previous_track_ids - current_track_ids
            
            for track_id in newly_lost_ids:
                if track_id not in lost_tracks:
                    lost_tracks[track_id] = {"history": track_history[track_id], "frames_lost": 1}
            
            # Update and predict for all lost tracks
            ghost_predictions = {}
            for track_id in list(lost_tracks.keys()):
                lost_tracks[track_id]["frames_lost"] += 1
                
                # Predict ghost track if history is sufficient
                history = lost_tracks[track_id]["history"]
                if len(history) >= INPUT_SEQUENCE_LENGTH:
                    input_seq = np.array(history)
                    input_seq_scaled = scaler.transform(input_seq)
                    input_tensor = torch.FloatTensor(input_seq_scaled).unsqueeze(0)
                    
                    with torch.no_grad():
                        predicted_seq_scaled = lstm_model(input_tensor)
                    
                    predicted_seq = scaler.inverse_transform(predicted_seq_scaled.squeeze(0).numpy())
                    ghost_predictions[track_id] = predicted_seq.astype(int)

            # --- ANNOTATION ---
            annotated_frame = frame.copy()
            # Draw ghost tracks
            for track_id, path in ghost_predictions.items():
                cv2.polylines(annotated_frame, [path], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Annotate active detections
            if detections.tracker_id is not None:
                labels = [f"{result.names[class_id]} ID:{tracker_id}" for _, _, _, class_id, tracker_id in detections]
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            cv2.imshow("Advanced Tracking with Ghosts", annotated_frame)
            sink.write_frame(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    print("Video processing completed.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_with_ghost_tracking()
