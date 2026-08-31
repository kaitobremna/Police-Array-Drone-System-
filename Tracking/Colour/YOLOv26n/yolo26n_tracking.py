from ultralytics import YOLO
import os

# 1. Load trained model
model = YOLO("/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Aerial image optimisation/YOLOv26n/yolo26n.pt")
# 2. Define path to your custom images !!CHANGE THIS!!
# Can be a single image, a folder, or even a video file
source_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/person14"
tracker_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Tracking/Colour/YOLOv26n/YOLOv26n_tracker.yaml"

# Output directory
output_dir = "runs/YOLO26n_tracks/person14_mot"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "results.txt")

# 3. Run Inference
print(f"🚀 Running model on: {source_path}")
results = model.track(
    source=source_path,
    show=True,
    stream=True,
    persist=True,
    tracker=tracker_path,         # Using Bot-sort with edited parameters for better long-term ID persistence
    save=True,                    # Save images with boxes drawn
    conf=0.163,                   # Confidence threshold (adjust if it misses things)
    #imgsz=640,                   # Processing resolution
    iou=0.5,                      # IoU threshold for tracking
    project="/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/runs/YOLO26n_tracks",
    name="person14"
)

with open(output_file, 'w') as f:
    frame_idx = 1
    
    for r in results:
        # Check if there are any detections in this frame
        if r.boxes.id is not None:
            # Get all data at once (CPU conversion)
            boxes_xywh = r.boxes.xywh.cpu().numpy()  # Center-X, Center-Y, W, H
            track_ids = r.boxes.id.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes_xywh):
                # Only track if it's a Person (Class 0) or Car (Class 1) if you wish
                # if int(cls[i]) != 0: continue 

                track_id = int(track_ids[i])
                confidence = confs[i]
                
                # CONVERT COORDINATES: Center -> Top-Left
                cx, cy, w, h = box
                x_top_left = cx - (w / 2)
                y_top_left = cy - (h / 2)

                # WRITE LINE: frame, id, x, y, w, h, conf, -1, -1, -1
                # (The -1s are placeholders for 3D coordinates required by the standard)
                line = f"{frame_idx},{track_id},{x_top_left:.2f},{y_top_left:.2f},{w:.2f},{h:.2f},{confidence:.2f},-1,-1,-1\n"
                f.write(line)
        
        # Progress update
        if frame_idx % 10 == 0:
            print(f"Processing Frame {frame_idx}...", end='\r')
            
        frame_idx += 1
print("✅ Done! Check the 'runs/YOLO26n_track/person14' folder.")