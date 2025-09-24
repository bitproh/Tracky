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

    # --- GPU DEVICE SETUP ---
    # Automatically detect and select the GPU (Quadro P2000), otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLO model and move it to the selected device
    model = YOLO(SOURCE_WEIGHTS_PATH)
    model.to(device)
    
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    # --- LOAD AND PREPARE LSTM ---
    lstm_model = TrajectoryLSTM()
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH))
    lstm_model.to(device) # Move LSTM model to the GPU
    lstm_model.eval()
    
    # Prepare a scaler to normalize and denormalize coordinates
    scaler = MinMaxScaler()
    scaler.fit(np.array([, [video_info.width, video_info.height]]))

    # --- TRACKING DATA STRUCTURES ---
    track_history = defaultdict(list)
    lost_tracks = {} # {tracker_id: { "frames_lost": 0 }}
    
    results_generator = model.track(
        source=SOURCE_VIDEO_PATH,
        tracker=TRACKER_CONFIG_PATH,
        conf=0.1, iou=0.7, stream=True, verbose=False,
        device=device # Explicitly tell the tracker to use the GPU
    )

    # Setup display window once before the loop
    window_name = "Advanced Tracking with Ghosts"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

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
                    
                    # --- THE FIX IS HERE ---
                    # Correctly calculate the center point of the bounding box
                    center_x = (bbox + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    track_history[tracker_id].append((center_x, center_y))
                    
                    # If this track was previously lost, remove it from the lost list
                    if tracker_id in lost_tracks:
                        del lost_tracks[tracker_id]

            # --- GHOST TRACKING LOGIC ---
            previous_track_ids = set(track_history.keys())
            newly_lost_ids = previous_track_ids - current_track_ids
            
            for track_id in newly_lost_ids:
                if track_id not in lost_tracks:
                    lost_tracks[track_id] = {"frames_lost": 1}
            
            ghost_predictions = {}
            for track_id in list(lost_tracks.keys()):
                lost_tracks[track_id]["frames_lost"] += 1
                
                history = track_history[track_id]
                if len(history) >= INPUT_SEQUENCE_LENGTH:
                    # Prepare the input sequence for the LSTM
                    input_seq = np.array(history)
                    input_seq_scaled = scaler.transform(input_seq)
                    input_tensor = torch.FloatTensor(input_seq_scaled).unsqueeze(0).to(device)
                    
                    # Predict the future trajectory
                    with torch.no_grad():
                        predicted_seq_scaled = lstm_model(input_tensor)
                    
                    # Convert the prediction back to pixel coordinates
                    predicted_seq = scaler.inverse_transform(predicted_seq_scaled.squeeze(0).cpu().numpy())
                    ghost_predictions[track_id] = predicted_seq.astype(int)

            # --- ANNOTATION ---
            annotated_frame = frame.copy()
            # Draw the yellow ghost tracks for lost objects
            for track_id, path in ghost_predictions.items():
                cv2.polylines(annotated_frame, [path], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Annotate the currently active detections
            if detections.tracker_id is not None:
                labels = [f"{result.names[class_id]} ID:{tracker_id}" for _, _, _, class_id, tracker_id in detections]
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            
            cv2.imshow(window_name, annotated_frame)
            sink.write_frame(annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    print("Video processing completed.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_with_ghost_tracking()
