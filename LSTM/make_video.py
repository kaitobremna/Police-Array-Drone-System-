import cv2
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from collections import defaultdict

# ==========================================
# 1. CONFIGURATION (UPDATE THESE PATHS!)
# ==========================================
# Path to the folder containing the .jpg images for ONE video sequence
IMG_FOLDER = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/person14"
# Path to the specific .txt annotation file for that sequence
ANNOTATION_FILE = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/runs/track_better/person14_mot/results.txt"

# Path to your trained model
MODEL_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/epoch_10.pth" 
OUTPUT_VIDEO = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/runs/predict/person14_prediction.mp4"

# ==========================================
# 2. MODEL DEFINITION (Must match training)
# ==========================================
class DroneTrajectoryLSTM(nn.Module):
    def __init__(self):
        super(DroneTrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :] 
        prediction = self.fc(last_step_out)
        return prediction

# ==========================================
# 3. PROCESSING LOADER
# ==========================================
def load_annotations(txt_file):
    # Organizes data by Frame ID -> Track ID -> Info
    # data[frame_id][track_id] = (x, y, w, h)
    data = defaultdict(dict)
    
    # We also need a global history for every track to feed the LSTM
    # history[track_id] = list of (cx_norm, cy_norm)
    full_tracks = defaultdict(list)
    
    with open(txt_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = parts[2], parts[3], parts[4], parts[5]
            
            # Center Point
            cx, cy = x + w/2, y + h/2
            
            data[frame_id][track_id] = (cx, cy, w, h)
            full_tracks[track_id].append((frame_id, cx, cy))
            
    return data, full_tracks

# ==========================================
# 4. MAIN GENERATOR
# ==========================================
def generate_video():
    # A. Load Model
    device = torch.device('cpu') # Mac runs on CPU usually
    model = DroneTrajectoryLSTM()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except:
        print("❌ Model not found. Check path.")
        return
    model.eval()

    # B. Load Annotations
    print("📂 Loading annotations...")
    frame_data, full_tracks_raw = load_annotations(ANNOTATION_FILE)
    
    # Pre-process tracks into dictionaries for fast lookup: history[track_id] = [(x,y), ...]
    # We need to build history dynamically as we loop frames to simulate real-time
    active_history = defaultdict(list) 

    # C. Setup Video Writer
    images = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
    if not images:
        print("❌ No images found! Check IMG_FOLDER path.")
        return

    # Read first frame to get size
    frame0 = cv2.imread(images[0])
    height, width, _ = frame0.shape
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    
    print(f"🎬 Generating Video ({len(images)} frames)...")

    # D. Loop Frames
    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        frame_id = i + 1 # VisDrone starts at frame 1 usually
        
        # Get all objects in this frame
        current_objects = frame_data[frame_id]
        
        # Overlay stats
        cv2.putText(frame, f"Frame: {frame_id}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        total_error = 0
        count = 0

        for track_id, (cx, cy, w, h) in current_objects.items():
            # 1. Update History
            # Normalize for LSTM (0-1)
            cx_norm, cy_norm = cx / 1920.0, cy / 1080.0
            active_history[track_id].append((cx_norm, cy_norm))
            
            # Keep only last 10
            if len(active_history[track_id]) > 10:
                active_history[track_id].pop(0)
            
            # 2. Draw Bounding Box (Ground Truth)
            top_left = (int(cx - w/2), int(cy - h/2))
            bottom_right = (int(cx + w/2), int(cy + h/2))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1) # Green Box
            
            # 3. RUN LSTM (If we have 10 frames of history)
            if len(active_history[track_id]) == 10:
                input_seq = torch.FloatTensor(active_history[track_id]).unsqueeze(0) # (1, 10, 2)
                
                with torch.no_grad():
                    pred = model(input_seq).squeeze(0) # (2,)
                
                # Un-normalize Prediction
                pred_x = int(pred[0].item() * 1920)
                pred_y = int(pred[1].item() * 1080)
                
                # 4. Draw The "Correction Line"
                # Line from Center -> Predicted Future
                start_point = (int(cx), int(cy))
                end_point = (pred_x, pred_y)
                
                # YELLOW LINE: The LSTM Vector
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 255), 2, tipLength=0.3)
                
                # RED DOT: The Predicted Location
                cv2.circle(frame, end_point, 4, (0, 0, 255), -1)

                # 5. Calculate Error (Future Lookahead)
                # Check if this object exists in the NEXT frame (frame_id + 1)
                if track_id in frame_data[frame_id + 1]:
                    true_next_cx, true_next_cy, _, _ = frame_data[frame_id + 1][track_id]
                    
                    # Error = Distance(Prediction, Actual Next)
                    error = np.sqrt((pred_x - true_next_cx)**2 + (pred_y - true_next_cy)**2)
                    total_error += error
                    count += 1
                    
                    # Draw a tiny red line showing the "Miss" distance
                    cv2.line(frame, end_point, (int(true_next_cx), int(true_next_cy)), (0, 0, 255), 1)

        # Show Avg Error on Screen
        if count > 0:
            avg_err = total_error / count
            cv2.putText(frame, f"Avg Correction Error: {avg_err:.1f} px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        if i % 50 == 0: print(f"Processed {i} frames...")

    out.release()
    print(f"✅ Done! Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    generate_video()