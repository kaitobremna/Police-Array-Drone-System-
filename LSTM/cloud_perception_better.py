import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import json
from collections import defaultdict, deque
from ultralytics import YOLO
from datetime import datetime, timezone

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Update your paths here
VIDEO_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/person14/%06d.jpg" 
LSTM_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/epoch_10.pth"
MODEL_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Aerial image optimisation/YOLOv8l/best (2).pt"
COMMAND_FILE = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/LSTM/Drone_commands/drone_trajectory.json"
TRACKER_PATH = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Tracking/Colour/drone_tracker.yaml"
LOOKAHEAD_STEPS = 5  # Predict 5 frames ahead

# Global Variables
track_history = defaultdict(lambda: deque(maxlen=10))

# ==========================================
# 2. SHARED DATA CONTAINER
# ==========================================
shared_data = {
    "boxes": [],
    "track_ids": [],
    "target_id": None,
}

# ==========================================
# 3. MOUSE CALLBACK
# ==========================================
def select_target(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        data = param
        print(f"\n🖱️ CLICK at ({x}, {y})")
        
        current_boxes = data["boxes"]
        current_ids = data["track_ids"]
        
        if len(current_boxes) == 0:
            print("⚠️ No objects found.")
            return

        # --- NEW STICKY LOGIC ---
        closest_dist = 99999
        closest_id = None
        
        for box, t_id in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = box
            
            # Calculate Center of this box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Calculate Distance from Mouse Click to Box Center
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            
            # If closer than current best, and within reasonable range (e.g. 150px)
            if dist < closest_dist and dist < 150: 
                closest_dist = dist
                closest_id = int(t_id)

        if closest_id is not None:
            data["target_id"] = closest_id
            print(f"✅ SNAPPED to Target #{closest_id} (Dist: {closest_dist:.1f}px)")
        else:
            print("❌ Click too far from any target.")
            data["target_id"] = None

# ==========================================
# 4. LSTM MODEL
# ==========================================
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

def predict_trajectory(model, history_seq, steps=LOOKAHEAD_STEPS):
    model.eval()
    current_seq = history_seq.clone()
    trajectory = []
    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_seq).unsqueeze(1)
            x, y = pred[0, 0, 0].item(), pred[0, 0, 1].item()
            trajectory.append((x, y))
            current_seq = torch.cat((current_seq[:, 1:, :], pred), dim=1)
    return trajectory

def send_packet(target_id, trajectory, w, h):
    current_time_unix = time.time()
    readable_time = datetime.fromtimestamp(current_time_unix, timezone.utc).strftime('%Y-%m-%d %H:%M:%S GMT')
    
    trajectory_data = []
    for i, (x, y) in enumerate(trajectory):
        yaw_val = x - 0.5
        pitch_val = -(y - 0.5) 
        trajectory_data.append({
            "step": int(i + 1),
            "time_offset": f"+{(i + 1) * 100}ms",
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "yaw_cmd": round(float(yaw_val), 4),
            "pitch_cmd": round(float(pitch_val), 4)
        })

    packet = {
        "packet_id": int(current_time_unix * 1000),
        "timestamp_readable": readable_time,
        "target_id": int(target_id),
        "box_size": {"w": round(float(w), 4), "h": round(float(h), 4)},
        "trajectory_commands": trajectory_data
    }
    
    with open(COMMAND_FILE, 'w') as f:
        json.dump(packet, f, indent=4)
    print(f"📡 SENT ID {target_id} | {readable_time}")

# ==========================================
# 5. MAIN LOOP (Final HUD Version)
# ==========================================
def run_system():
    yolo = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Window Setup
    cv2.namedWindow("HUD", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("HUD", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Uncomment for fullscreen
    cv2.resizeWindow("HUD", 1920, 1080)
    
    cv2.setMouseCallback("HUD", select_target, param=shared_data)

    print("🚀 System Ready.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # 1. Run Tracker
        results = yolo.track(frame, persist=True, tracker=TRACKER_PATH, verbose=False)
        
        if results[0].boxes.id is not None:
            # Update Shared Data
            shared_data["boxes"] = results[0].boxes.xyxy.cpu().numpy()
            shared_data["track_ids"] = results[0].boxes.id.int().cpu().numpy()
            
            # --- HUD: OBJECT COUNTER ---
            num_objects = len(shared_data["track_ids"])
            status_text = "SEARCHING"
            status_color = (0, 255, 255) # Yellow
            
            if shared_data["target_id"] is not None:
                status_text = f"TRACKING ID #{shared_data['target_id']}"
                status_color = (0, 255, 0) # Green

            # Draw Top-Left Stats
            # Shadow (Black) for readability
            cv2.putText(frame, f"Objects Detected: {num_objects}", (22, 52), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, f"System: {status_text}", (22, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            
            # Text (Colored)
            cv2.putText(frame, f"Objects Detected: {num_objects}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"System: {status_text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # --- DRAW BOXES ---
            for box, track_id in zip(shared_data["boxes"], shared_data["track_ids"]):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                track_history[track_id].append((cx / 1920.0, cy / 1080.0))
                
                is_target = (shared_data["target_id"] is not None and track_id == shared_data["target_id"])
                
                if is_target:
                    # GREEN BOX
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    if len(track_history[track_id]) == 10:
                        input_seq = torch.FloatTensor(list(track_history[track_id])).unsqueeze(0)
                        traj = predict_trajectory(lstm_model, input_seq, steps=LOOKAHEAD_STEPS)
                        
                        w, h = (x2-x1)/1920, (y2-y1)/1080
                        send_packet(track_id, traj, w, h)
                        
                        # Arrow & Dot
                        final_x, final_y = traj[-1] 
                        pixel_start = (int(cx), int(cy))
                        pixel_end = (int(final_x * 1920), int(final_y * 1080))
                        
                        cv2.arrowedLine(frame, pixel_start, pixel_end, (0, 255, 255), 2, tipLength=0.3)
                        cv2.circle(frame, pixel_end, 4, (0, 0, 255), -1)

                else:
                    # GREY BOX
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (100, 100, 100), 1)

        cv2.imshow("HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()