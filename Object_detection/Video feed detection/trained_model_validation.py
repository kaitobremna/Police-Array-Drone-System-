from ultralytics import YOLO

# 1. Load your trained model
# Replace with the path to the 'best.pt' file you downloaded from Kaggle/Colab
model = YOLO("/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/Object_detection/Aerial image optimisation/best.pt") 

# 2. Define path to your custom images !!CHANGE THIS!!
# Can be a single image, a folder, or even a video file
source_path = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/UAV123_10fps/data_seq/UAV123_10fps/person1_s"

# 3. Run Inference
print(f"🚀 Running model on: {source_path}")
results = model.predict(
    source=source_path,
    stream=True,
    save=True,          # Save images with boxes drawn
    conf=0.4,           # Confidence threshold (adjust if it misses things)
    imgsz=640,          # Processing resolution
    show=True,
    project="runs/detect",
    name="video_test"
)

# 4. TRIGGER THE PROCESSING (The missing part)
# The model only processes a frame when the loop asks for it.
for result in results:
    pass # You don't need to do anything inside; the loop itself keeps it running.

print("✅ Done! Check the 'runs/detect/video_test' folder.")