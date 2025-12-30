from ultralytics import YOLO
import os

# 1. Load the normal, pre-trained model
model = YOLO('yolov8n.pt') 

# 2. Run Inference (Not Validation)
# We use 'predict' to see what the model "sees", ignoring the class mismatch errors.
# save=True will generate the images with boxes drawn.
# conf=0.25 is the standard threshold.
print("📸 Taking snapshots with standard model...")

# Note: We point to the dataset folder that was downloaded earlier
# Adjust this path if your dataset is elsewhere (e.g. /content/datasets/VisDrone/...)
results = model.predict(
    source='/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Visdrone sample images', 
    conf=0.25, 
    save=True, 
    imgsz=640,  # Standard size (shows why resizing hurts small objects)
    project='runs/detect',
    name='baseline_visuals',
    max_det=100  # Limit boxes so it doesn't crash on crowds
)

print(f"✅ Baseline complete. Check the images in 'runs/detect/baseline_visuals'")