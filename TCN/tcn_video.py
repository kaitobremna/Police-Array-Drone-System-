import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
from collections import defaultdict

# ==========================================
# 1. CONFIGURATION (UPDATE THESE PATHS!)
# ==========================================
# Path to the folder containing the .jpg images for ONE video sequence
IMG_FOLDER = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/person14"
ANNOTATION_FILE = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/runs/track_better/person14_mot/results.txt"

# ⚠️ UPDATE TO YOUR TCN MODEL PATH
MODEL_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/TCN/best_tcn_resnet.pth" 
OUTPUT_VIDEO = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/runs/predict/person14_TCN_prediction.mp4"

# ==========================================
# 2. MODEL DEFINITION (The Winning TCN)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation
        out = F.pad(x, (pad, 0))
        out = F.relu(self.conv1(out))
        out = F.pad(out, (pad, 0))
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class DroneTrajectoryTCN_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(in_channels=2, out_channels=32, kernel_size=3, dilation=1)
        self.block2 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=2)
        self.block3 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=4)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = x[:, :, -1] 
        return self.fc(out)

# ==========================================
# 3. PROCESSING LOADER
# ==========================================
def load_annotations(txt_file):
    data = defaultdict(dict)
    full_tracks = defaultdict(list)
    
    with open(txt_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split(',')))
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = parts[2], parts[3], parts[4], parts[5]
            
            cx, cy = x + w/2, y + h/2
            data[frame_id][track_id] = (cx, cy, w, h)
            full_tracks[track_id].append((frame_id, cx, cy))
            
    return data, full_tracks

# ==========================================
# 4. MAIN GENERATOR
# ==========================================
def generate_video():
    # A. Load Model
    device = torch.device('cpu') 
    model = DroneTrajectoryTCN_ResNet()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except:
        print("❌ Model not found. Check MODEL_PATH.")
        return
    model.eval()

    # B. Load Annotations
    print("📂 Loading annotations...")
    frame_data, full_tracks_raw = load_annotations(ANNOTATION_FILE)
    active_history = defaultdict(list) 

    # C. Setup Video Writer
    images = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
    if not images:
        print("❌ No images found! Check IMG_FOLDER path.")
        return

    frame0 = cv2.imread(images[0])
    height, width, _ = frame0.shape
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    
    print(f"🎬 Generating TCN Video ({len(images)} frames)...")

    # D. Loop Frames
    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        frame_id = i + 1 
        current_objects = frame_data[frame_id]
        
        cv2.putText(frame, f"Frame: {frame_id} | TCN Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        total_error = 0
        count = 0

        for track_id, (cx, cy, w, h) in current_objects.items():
            # 1. Update History (Normalize to 0-1)
            cx_norm, cy_norm = cx / 1920.0, cy / 1080.0
            active_history[track_id].append((cx_norm, cy_norm))
            
            # Keep only last 30 for the TCN
            if len(active_history[track_id]) > 30:
                active_history[track_id].pop(0)
            
            # 2. Draw Bounding Box
            top_left = (int(cx - w/2), int(cy - h/2))
            bottom_right = (int(cx + w/2), int(cy + h/2))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1) 
            
            # 3. RUN TCN
            # Use Cold Start Padding so we predict instantly on frame 1
            current_history = list(active_history[track_id])
            while len(current_history) < 30:
                current_history.insert(0, current_history[0])
                
            input_seq = torch.FloatTensor(current_history).unsqueeze(0) # (1, 30, 2)
            
            with torch.no_grad():
                pred = model(input_seq).squeeze(0) # (2,)
            
            # Un-normalize Prediction
            pred_x = int(pred[0].item() * 1920)
            pred_y = int(pred[1].item() * 1080)
            
            # 4. Draw The "Correction Line"
            start_point = (int(cx), int(cy))
            end_point = (pred_x, pred_y)
            
            # YELLOW LINE: The TCN Vector
            cv2.arrowedLine(frame, start_point, end_point, (0, 255, 255), 2, tipLength=0.3)
            # RED DOT: The Predicted Location
            cv2.circle(frame, end_point, 4, (0, 0, 255), -1)

            # 5. Calculate Error (Future Lookahead)
            if track_id in frame_data[frame_id + 1]:
                true_next_cx, true_next_cy, _, _ = frame_data[frame_id + 1][track_id]
                
                error = np.sqrt((pred_x - true_next_cx)**2 + (pred_y - true_next_cy)**2)
                total_error += error
                count += 1
                
                # Draw a tiny red line showing the "Miss" distance
                cv2.line(frame, end_point, (int(true_next_cx), int(true_next_cy)), (0, 0, 255), 1)

        if count > 0:
            avg_err = total_error / count
            cv2.putText(frame, f"Avg TCN Pixel Error: {avg_err:.1f} px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        if i % 50 == 0: print(f"Processed {i} frames...")

    out.release()
    print(f"✅ Done! Video saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    generate_video()