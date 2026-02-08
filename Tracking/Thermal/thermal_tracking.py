from ultralytics import YOLO

# 1. Load trained model
model = YOLO("/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Thermal image Optimisation/best.pt") 

# 2. Define path to your custom images !!CHANGE THIS!!
# Can be a single image, a folder, or even a video file
source_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/Thermal Drone footage/cops_thermal.f134.mov"
tracker_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Tracking/Thermal/thermal_tracker.yaml"


# 3. Run Inference
print(f"🚀 Running model on: {source_path}")
results = model.track(
    source=source_path,
    stream=True,
    persist=True,
    tracker=tracker_path,         # Using Bot-sort with edited parameters for better long-term ID persistence
    save=True,                    # Save images with boxes drawn
    conf=0.25,                    # Confidence threshold (adjust if it misses things)
    imgsz=640,                    # Processing resolution
    device='mps',                 # Use 'mps' for Mac with M1/M2 chips; use 'cuda' for Nvidia GPUs; use 'cpu' if no GPU
    show=True,
    iou=0.5,                      # IoU threshold for tracking
    project="runs/track_thermal",
    name="cops_thermal"
)

# 4. TRIGGER THE PROCESSING 
# The model only processes a frame when the loop asks for it.
for result in results:
    pass 

print("✅ Done! Check the 'runs/track_thermal/cops_thermal' folder.")