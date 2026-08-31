import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from torch import nn
import torch.nn.functional as F

class DroneTrajectoryLSTM(nn.Module):
    def __init__(self):
        super(DroneTrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Two causal convolutions per block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        
        # Safety valve: 1x1 conv if channel sizes change
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Compute left padding to ensure STRICT causality (no future leaks)
        pad = (self.kernel_size - 1) * self.dilation

        # 1st Conv: Pad left, convolve, activate
        out = F.pad(x, (pad, 0))
        out = F.relu(self.conv1(out))

        # 2nd Conv: Pad left, convolve
        out = F.pad(out, (pad, 0))
        out = self.conv2(out)

        # Residual Skip Connection (Add original input back to the output)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)
    
class DroneTrajectoryTCN_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Exponentially growing dilations: 1, 2, 4
        self.block1 = ResidualBlock(in_channels=2, out_channels=32, kernel_size=3, dilation=1)
        self.block2 = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=2)
        self.block3 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, dilation=4)
        
        # Final fully connected layer to output the next (x, y) coordinate
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # Flip shape from [Batch, Seq=30, Features=2] to [Batch, Features=2, Seq=30]
        x = x.permute(0, 2, 1) 
        
        # Pass through the residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Extract the feature vector of the final timestep
        out = x[:, :, -1] 
        
        # Predict the next (x, y)
        return self.fc(out)
    
def load_visdrone_trajectories(annotation_dir, seq_length=30, img_w=1920, img_h=1080):
    """
    Parses VisDrone MOT .txt files into sequential training/validation tensors.
    """
    print(f"⚙️ Parsing VisDrone tracking data from: {annotation_dir}")
    X, y = [], []
    
    # VisDrone Categories: 1:pedestrian, 2:people, 4:car, 5:van, 6:truck, 9:bus
    valid_categories = {1, 2, 4, 5, 6, 9}
    total_tracks_extracted = 0

    for filename in os.listdir(annotation_dir):
        if not filename.endswith('.txt'): continue
        filepath = os.path.join(annotation_dir, filename)
        
        # Dictionary to hold the trajectory of each unique ID in this video
        tracks = defaultdict(list)
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                
                # VisDrone MOT format: frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category...
                frame_idx = int(parts[0])
                target_id = int(parts[1])
                left, top, w, h = map(float, parts[2:6])
                category = int(parts[7])
                
                # Filter out irrelevant objects (like bicycles or ignored regions)
                if category not in valid_categories: 
                    continue
                
                # Calculate Center X and Center Y, and normalize to [0, 1]
                cx = (left + (w / 2.0)) / img_w
                cy = (top + (h / 2.0)) / img_h
                
                tracks[target_id].append((frame_idx, cx, cy))
        
        # Chop the continuous tracks into sliding windows
        for target_id, positions in tracks.items():
            # Sort chronologically just in case the text file is out of order
            positions.sort(key=lambda item: item[0])
            coords = np.array([[p[1], p[2]] for p in positions])
            
            # If the car was tracked for less than the sequence length, ignore it
            if len(coords) <= seq_length:
                continue
            
            total_tracks_extracted += 1
            
            # Sliding window: take 29 frames as input, use the 30th as the target
            for i in range(len(coords) - seq_length):
                seq = coords[i : i + seq_length - 1]  # The historical sequence
                target = coords[i + seq_length - 1]   # The future step to predict
                
                X.append(seq)
                y.append(target)

    print(f"✅ Extracted {len(X)} sequences from {total_tracks_extracted} unique object tracks.")
    
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

def visualize_random_trajectory(tcn_path, lstm_path, val_dir):
    print("🎨 GENERATING TRAJECTORY VISUALIZATION 🎨\n")
    
    # We can run this on CPU since it's just one single prediction
    device = torch.device("cpu")
    
    # 1. Load Data
    X_val, y_val = load_visdrone_trajectories(val_dir, seq_length=30)
    
    # Pick a random sequence
    idx = random.randint(0, len(X_val) - 1)
    seq = X_val[idx].unsqueeze(0).to(device) # Shape: [1, 30, 2]
    true_target = y_val[idx].to(device)      # Shape: [2]
    
    # 2. Initialize Models
    tcn_model = DroneTrajectoryTCN_ResNet().to(device)
    lstm_model = DroneTrajectoryLSTM().to(device)
    
    tcn_model.load_state_dict(torch.load(tcn_path, map_location=device))
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    
    tcn_model.eval()
    lstm_model.eval()
    
    # 3. Get Predictions
    with torch.no_grad():
        # TCN gets all 30 frames
        tcn_pred = tcn_model(seq).squeeze(0)
        
        # LSTM gets only the last 10 frames
        lstm_seq = seq[:, -10:, :]
        lstm_pred = lstm_model(lstm_seq).squeeze(0)
        
    # 4. Extract Coordinates for Plotting
    # Historical path (using all 30 frames for visual context)
    hist_x = seq[0, :, 0].numpy()
    hist_y = seq[0, :, 1].numpy()
    
    # Targets and Guesses
    true_x, true_y = true_target[0].item(), true_target[1].item()
    tcn_px, tcn_py = tcn_pred[0].item(), tcn_pred[1].item()
    lstm_px, lstm_py = lstm_pred[0].item(), lstm_pred[1].item()
    
    # 5. Build the Graph
    plt.figure(figsize=(10, 8))
    
    # Plot the drone's 3-second historical tracking path
    plt.plot(hist_x, hist_y, marker='o', color='gray', linestyle='--', alpha=0.6, label="Drone's Past Tracking (30 frames)")
    
    # Highlight the very last known position before predicting
    plt.scatter(hist_x[-1], hist_y[-1], color='black', s=100, label="Last Known Position", zorder=5)
    
    # Plot the True Future Target (Ground Truth)
    plt.scatter(true_x, true_y, color='green', marker='*', s=300, label="True Next Position (Target)", zorder=5)
    
    # Plot TCN Prediction
    plt.scatter(tcn_px, tcn_py, color='blue', marker='X', s=200, label="TCN Prediction (Winner)", zorder=5)
    
    # Plot LSTM Prediction
    plt.scatter(lstm_px, lstm_py, color='red', marker='P', s=200, label="LSTM Prediction", zorder=5)
    
    # Formatting
    plt.title("TCN vs LSTM: Autonomous Trajectory Prediction", fontsize=16, fontweight='bold')
    plt.xlabel("Normalized X Coordinate", fontsize=12)
    plt.ylabel("Normalized Y Coordinate", fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Invert Y axis because image coordinates (0,0) start at the top-left
    plt.gca().invert_yaxis() 
    
    plt.show()

# --- HOW TO RUN IT ---
LSTM_DIR = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/epoch_10.pth"
TCN_DIR = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/TCN/best_tcn_resnet.pth"
VAL_DIR = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/DoneVDT18/VisDrone2019-VID-val/annotations"
visualize_random_trajectory(TCN_DIR, LSTM_DIR, VAL_DIR)