import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

# ==========================================
# 1. CONFIGURATION (Matches Paper Section 4.3)
# ==========================================
CONFIG = {
    'input_size': 2,      # (x, y)
    'hidden_size': 64,    # 
    'num_layers': 1,      # Simple architecture
    'output_size': 2,     # Predict next (x, y)
    'seq_length': 10,     # Past 10 frames 
    'batch_size': 32,     # [cite: 457]
    'learning_rate': 0.001, # Starting LR 
    'epochs': 150,        # [cite: 454]
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==========================================
# 2. THE MODEL (Section 3.2)
# ==========================================
class DroneTrajectoryLSTM(nn.Module):
    def __init__(self):
        super(DroneTrajectoryLSTM, self).__init__()
        # The exact architecture described in the paper
        self.lstm = nn.LSTM(input_size=CONFIG['input_size'],
                            hidden_size=CONFIG['hidden_size'],
                            num_layers=CONFIG['num_layers'],
                            batch_first=True)
        
        # Simple Linear layer to map 64 features -> 2 coordinates
        self.fc = nn.Linear(CONFIG['hidden_size'], CONFIG['output_size'])

    def forward(self, x):
        # x shape: (Batch, Sequence_Len, 2)
        
        # 1. LSTM Layer
        # out shape: (Batch, Seq_Len, 64)
        lstm_out, _ = self.lstm(x)
        
        # 2. Extract Last Time Step
        # We only care about the state after seeing the 10th frame
        last_step_out = lstm_out[:, -1, :] 
        
        # 3. Prediction
        prediction = self.fc(last_step_out)
        return prediction

# ==========================================
# 3. DATASET HANDLING (VisDrone Style)
# ==========================================
class VisDroneTrajectoryDataset(Dataset):
    def __init__(self, data_dir, seq_length=10):
        """
        data_dir: Path to the folder containing your .txt files (e.g., "VisDrone2018-VDT-Train/annotations")
        """
        self.seq_length = seq_length
        self.samples = []
        
        # 1. Get all text files in the folder
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        print(f"📂 Found {len(txt_files)} annotation files. Parsing...")

        for txt_file in txt_files:
            # Temporary dict to group data by Track ID
            # tracks[track_id] = list of (x, y)
            tracks = {}
            
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split(',')))
                    
                    # Parse Columns
                    frame_id = parts[0]
                    track_id = int(parts[1])
                    x, y, w, h = parts[2], parts[3], parts[4], parts[5]
                    
                    # 2. CONVERT TO CENTER POINTS
                    # The LSTM predicts the center of the object, not the top-left corner
                    cx = x + (w / 2)
                    cy = y + (h / 2)
                    
                    # 3. NORMALIZE (CRITICAL!)
                    # Assuming 1920x1080 resolution (Standard VisDrone)
                    # If you don't do this, the Loss will be huge (e.g., 2000.0)
                    cx_norm = cx / 1920.0
                    cy_norm = cy / 1080.0
                    
                    if track_id not in tracks:
                        tracks[track_id] = []
                    tracks[track_id].append((cx_norm, cy_norm))
            
            # 4. Create Sliding Windows (Sequences)
            # We need sequences of length 11 (10 Input + 1 Target)
            for t_id, points in tracks.items():
                if len(points) > seq_length:
                    for i in range(len(points) - seq_length):
                        # Input: 10 frames
                        inp = points[i : i + seq_length]
                        # Target: The 11th frame
                        target = points[i + seq_length]
                        
                        self.samples.append((inp, target))

        print(f"✅ Loaded {len(self.samples)} training samples from {len(txt_files)} videos.")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, target = self.samples[idx]
        # Convert to Tensor (Float32 is standard for AI)
        return torch.FloatTensor(inp), torch.FloatTensor(target)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    print(f"🚀 Training on {CONFIG['device']} with Batch Size {CONFIG['batch_size']}")
    
    # 1. Setup Data
    train_dataset = VisDroneTrajectoryDataset(data_dir="/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/DoneVDT18/VisDrone2019-VID-train/annotations")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 2. Setup Model
    model = DroneTrajectoryLSTM().to(CONFIG['device'])
    
    # 3. Optimizer & Loss [cite: 451]
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    # 4. Scheduler (Decay) 
    # "Decaying from 0.001 to 10^-5" over 150 epochs
    # Gamma calculation: (1e-5 / 1e-3)^(1/150) ≈ 0.97
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # 5. Loop
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            # Forward Pass
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Step the scheduler
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")

    # 6. Save
    print("✅ Training Complete.")
    torch.save(model.state_dict(), "drone_lstm_correction.pth")
    print("💾 Model saved to 'drone_lstm_correction.pth'")

if __name__ == "__main__":
    train()