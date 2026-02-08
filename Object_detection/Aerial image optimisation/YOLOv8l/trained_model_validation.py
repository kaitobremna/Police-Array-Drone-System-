from ultralytics import YOLO

# 1. Load your trained model
# Replace with the path to the 'best.pt' file you downloaded from Kaggle/Colab
model = YOLO("/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Aerial image optimisation/YOLOv8l/best (2).pt") 

# 2. Define path to your custom images
# Can be a single image, a folder, or even a video file
source_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Visdrone sample images"

# 3. Run Inference
print(f"🚀 Running model on: {source_path}")
results = model.predict(
    source=source_path,
    save=True,          # Save images with boxes drawn
    conf=0.4,           # Confidence threshold (adjust if it misses things)
    imgsz=640,          # Processing resolution
    project="runs/detect",
    name="visual_test"
)

print("✅ Done! Check the 'runs/detect/visual_test' folder.")