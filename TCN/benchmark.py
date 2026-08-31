import torch
from torch import nn
import time
import torch.nn.functional as F


#CONFIG 
TCN_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/TCN/best_tcn_resnet.pth"
LSTM_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/best_drone_lstm_model.pth"

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

class DroneTrajectoryLSTM(nn.Module):
    def __init__(self):
        super(DroneTrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

device = 'cpu'
lstm_model = DroneTrajectoryLSTM()
try:
    lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    lstm_model.eval()
    print("✅ LSTM Model Loaded.")
except:
    print("❌ LSTM Model Not Found.")
    
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
    
def compare_models(tcn_path, lstm_path):
    print("🥊 STARTING HEAD-TO-HEAD BENCHMARK 🥊\n")
    
    # Force to CPU to simulate your drone's edge computer!
    device = torch.device("cpu") 
    
    # 1. Initialize Both Models
    tcn_model = DroneTrajectoryTCN_ResNet().to(device)
    lstm_model = DroneTrajectoryLSTM().to(device) 
    
    # Load weights (assuming you have both .pth files)
    tcn_model.load_state_dict(torch.load(tcn_path, map_location=device))
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    
    tcn_model.eval()
    lstm_model.eval()
    
    # 2. Compare Model Size (Memory Footprint)
    tcn_params = sum(p.numel() for p in tcn_model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    
    print("📊 1. MEMORY FOOTPRINT (Parameter Count)")
    print(f"   TCN:  {tcn_params:,} parameters")
    print(f"   LSTM: {lstm_params:,} parameters\n")

    # 3. Compare Inference Speed (Latency)
    # Create dummy tensors representing ONE bounding box sequence of 30 frames
    dummy_tcn = torch.randn(1, 30, 2).to(device) 
    dummy_lstm = torch.randn(1, 10, 2).to(device) 
    
    # Warm up the CPU (PyTorch is slow on the very first run)
    for _ in range(10):
        _ = tcn_model(dummy_tcn)
        _ = lstm_model(dummy_lstm)

    print("⚡ 2. INFERENCE SPEED (Time to predict next frame)")
    
    # Time TCN
    start_time = time.perf_counter()
    for _ in range(1000): # Run 1000 times for a stable rage
        _ = tcn_model(dummy_tcn)
    tcn_time = (time.perf_counter() - start_time) / 1000
    print(f"   TCN:  {tcn_time * 1000:.4f} milliseconds per prediction")
    
    # Time LSTM
    start_time = time.perf_counter()
    for _ in range(1000):
        _ = lstm_model(dummy_lstm)
    lstm_time = (time.perf_counter() - start_time) / 1000
    print(f"   LSTM: {lstm_time * 1000:.4f} milliseconds per prediction\n")
    
    print("🎯 3. ACCURACY")
    print("   (Run your validation loop on both models to compare their Val Loss!)")

# Run the benchmark
if __name__ == "__main__":
    compare_models(TCN_PATH, LSTM_PATH)