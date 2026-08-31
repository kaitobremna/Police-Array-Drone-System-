import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import json
from collections import defaultdict, deque
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
VIDEO_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/person14/%06d.jpg"   
LSTM_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/epoch_10.pth"     
COMMAND_FILE = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/Drone_commands/person13_drone_command.json"
LOOKAHEAD_STEPS = 4                     
TRACKER_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Tracking/Colour/drone_tracker.yaml"
MODEL_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Aerial image optimisation/YOLOv8l/best (2).pt"

# Global Variables for "Target Lock"
TARGET_ID = None
track_history = defaultdict(lambda: deque(maxlen=10))

# ==========================================
# 2. LSTM MODEL DEFINITION
# ==========================================
class DroneTrajectoryLSTM(nn.Module):
    def __init__(self):
        super(DroneTrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load Model
device = 'cpu'
lstm_model = DroneTrajectoryLSTM()
try:
    lstm_model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    lstm_model.eval()
    print("✅ LSTM Model Loaded Successfully.")
except:
    print("⚠️ WARNING: LSTM Model not found. Running in Visual-Only Mode.")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def predict_multistep(model, history_seq, steps=4):
    """Recursively predicts 'steps' frames ahead."""
    model.eval()
    current_seq = history_seq.clone()
    final_prediction = None
    
    for _ in range(steps):
        with torch.no_grad():
            pred = model(current_seq).unsqueeze(1)
            final_prediction = pred.squeeze(0).squeeze(0)
            current_seq = torch.cat((current_seq[:, 1:, :], pred), dim=1)
            
    return final_prediction # Returns (x, y) normalized

def send_drone_command(target_id, pred_x_norm, pred_y_norm):
    """Calculates Yaw/Pitch vectors and saves JSON file."""
    # Center is 0.5. Range is -0.5 (Left/Top) to +0.5 (Right/Bottom)
    vector_x = pred_x_norm - 0.5
    vector_y = pred_y_norm - 0.5 
    
    # Invert Y for Pitch (Top of screen = Fly Forward = Positive Pitch)
    pitch_cmd = -vector_y 
    yaw_cmd = vector_x

    command_packet = {
        "packet_id": int(time.time() * 1000),
        "timestamp": time.time(),
        "target_id": int(target_id),
        "prediction": {
            "horizon_frames": LOOKAHEAD_STEPS,
            "screen_x": round(float(pred_x_norm), 4),
            "screen_y": round(float(pred_y_norm), 4)
        },
        "control_vector": {
            "yaw": round(float(yaw_cmd), 4),     # +Right / -Left
            "pitch": round(float(pitch_cmd), 4)  # +Forward / -Back
        }
    }
    
    # Save to file (Simulating Network Transmission)
    with open(COMMAND_FILE, 'w') as f:
        json.dump(command_packet, f, indent=4)

def select_target(event, x, y, flags, param):
    """Mouse Callback to Lock/Unlock Targets."""
    global TARGET_ID
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes, track_ids = param
        found = False
        for box, t_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                TARGET_ID = int(t_id)
                print(f"🎯 TARGET LOCKED: ID #{TARGET_ID}")
                found = True
                break
        if not found:
            print("❌ Target Lost (Clicked Empty Space)")
            TARGET_ID = None

# ==========================================
# 4. MAIN LOOP
# ==========================================
def run_system():
    

    yolo = YOLO(MODEL_PATH) 
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow("Cloud Perception HUD")
    cv2.setMouseCallback("Cloud Perception HUD", select_target)

    print("🚀 System Running... Click a vehicle to Track & Control!")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # A. Run Tracking
        results = yolo.track(frame, persist=True, tracker=TRACKER_PATH, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            
            # Update Mouse Callback Data
            cv2.setMouseCallback("Cloud Perception HUD", select_target, param=(boxes, track_ids))

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # B. Update History (Always do this for everyone)
                track_history[track_id].append((cx / 1920.0, cy / 1080.0))
                
                # Default Colors (Gray for ignored objects)
                color = (100, 100, 100)
                thickness = 1
                
                # C. TARGET LOGIC (Only if ID matches Locked Target)
                if TARGET_ID is not None and track_id == TARGET_ID:
                    color = (0, 255, 0) # Green
                    thickness = 3
                    
                    # Run LSTM if we have enough history
                    if len(track_history[track_id]) == 10:
                        input_seq = torch.FloatTensor(list(track_history[track_id])).unsqueeze(0)
                        
                        # PREDICT 4 STEPS AHEAD
                        future_pred = predict_multistep(lstm_model, input_seq, steps=LOOKAHEAD_STEPS)
                        
                        pred_x = future_pred[0].item()
                        pred_y = future_pred[1].item()
                        
                        # SEND COMMAND
                        send_drone_command(track_id, pred_x, pred_y)

                        # VISUALIZE
                        pixel_x, pixel_y = int(pred_x * 1920), int(pred_y * 1080)
                        
                        # Yellow Arrow (Current -> Future)
                        cv2.arrowedLine(frame, (int(cx), int(cy)), (pixel_x, pixel_y), (0, 255, 255), 4, tipLength=0.3)
                        # Red Dot (Future Position)
                        cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 0, 255), -1)
                        
                        # HUD Text
                        vector_x = pred_x - 0.5
                        cv2.putText(frame, f"LOCKED #{track_id}", (int(x1), int(y1)-30), 0, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"YAW CMD: {vector_x:.2f}", (int(x1), int(y1)-10), 0, 0.6, (0, 255, 255), 2)

                # Draw Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        cv2.imshow("Cloud Perception HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()