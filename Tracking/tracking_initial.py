from ultralytics import YOLO

# 1. Load trained model
model = YOLO("/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Aerial image optimisation/best.pt") 

# 2. Define path to your custom images !!CHANGE THIS!!
# Can be a single image, a folder, or even a video file
source_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/bike1"
tracker_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Tracking/drone_tracker.yaml"

# 3. Run Inference
print(f"🚀 Running model on: {source_path}")
results = model.track(
    source=source_path,
    stream=True,
    persist=True,
    tracker=tracker_path,         # Using Bot-sort with edited parameters for better long-term ID persistence
    save=True,                    # Save images with boxes drawn
    conf=0.4,                     # Confidence threshold (adjust if it misses things)
    imgsz=640,                    # Processing resolution
    show=True,
    iou=0.5,                      # IoU threshold for tracking
    project="runs/track",
    name="bike1"
)

# 4. TRIGGER THE PROCESSING 
# The model only processes a frame when the loop asks for it.
for result in results:
    pass 

print("✅ Done! Check the 'runs/track/bike1' folder.")