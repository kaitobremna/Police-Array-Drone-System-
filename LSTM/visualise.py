import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
import os
import random
import numpy as np

# ==========================================
# 1. SETUP (Point to your files here!)
# ==========================================
# PATH TO YOUR DOWNLOADED MODEL
MODEL_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/epoch_10.pth" 

# PATH TO YOUR LOCAL DATA (Check your first screenshot for this path)
DATA_DIR = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/DoneVDT18/VisDrone2019-VID-val/annotations" 

# ==========================================
# 2. DEFINE THE MODEL (Must match training exactly)
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
# 3. DEFINE THE DATA LOADER
# ==========================================
class VisDroneTrajectoryDataset(Dataset):
    def __init__(self, data_dir, seq_length=10):
        self.samples = []
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        
        # Limit to 5 files for speed if just testing, or remove limit for full set
        txt_files = txt_files[:5] 
        print(f"📂 Parsing {len(txt_files)} local files...")

        for txt_file in txt_files:
            tracks = {}
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split(',')))
                    track_id = int(parts[1])
                    # Center Point Calculation
                    cx = parts[2] + (parts[4] / 2)
                    cy = parts[3] + (parts[5] / 2)
                    # Normalize
                    cx_norm, cy_norm = cx / 1920.0, cy / 1080.0
                    
                    if track_id not in tracks: tracks[track_id] = []
                    tracks[track_id].append((cx_norm, cy_norm))
            
            for t_id, points in tracks.items():
                if len(points) > seq_length:
                    for i in range(len(points) - seq_length):
                        self.samples.append((points[i:i+seq_length], points[i+seq_length]))
        print(f"✅ Loaded {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx][0]), torch.FloatTensor(self.samples[idx][1])

# ==========================================
# 4. VISUALIZATION FUNCTION
# ==========================================
def run_visualizer():
    # A. Load Data
    dataset = VisDroneTrajectoryDataset(DATA_DIR)
    
    # B. Load Model
    print("🧠 Loading Model...")
    model = DroneTrajectoryLSTM()
    
    # CRITICAL: This loads the GPU model onto your CPU/Mac
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"❌ Error: Could not find model at {MODEL_PATH}")
        return

    model.eval()
    
    # C. Pick a random sample and predict
    idx = random.randint(0, len(dataset)-1)
    input_seq, target = dataset[idx]
    
    with torch.no_grad():
        pred = model(input_seq.unsqueeze(0)).squeeze(0)

    # D. Un-normalize and Plot
    W, H = 1920, 1080
    hist_x = input_seq[:, 0].numpy() * W
    hist_y = input_seq[:, 1].numpy() * H
    true_x, true_y = target[0].item() * W, target[1].item() * H
    pred_x, pred_y = pred[0].item() * W, pred[1].item() * H

    plt.figure(figsize=(10, 6))
    plt.plot(hist_x, hist_y, 'bo-', label='Past Path')
    plt.scatter(true_x, true_y, c='g', s=150, marker='*', label='Actual Future')
    plt.scatter(pred_x, pred_y, c='r', s=150, marker='x', label='LSTM Prediction')
    plt.legend()
    plt.title(f"Local Visualization (Sample #{idx})")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_visualizer()