from ultralytics import YOLO

# 1. Load the P2 architecture (builds from scratch)
model = YOLO("yolov8-p2.yaml") 

# 2. Transfer weights from standard YOLOv8n (optional but recommended for speed)
# This loads the weights that match, skipping the new P2 layers which will be learned.
model = model.load("yolov8n.pt") 

# # 3. Train with Aerial-Specific Hyperparameters
# results = model.train(
#     data="VisDrone.yaml",  # Standard VisDrone dataset config
#     epochs=100,            # 100 is a good baseline for P2 convergence
#     imgsz=640,             # Keep 640 for speed; P2 layer handles the resolution detail
#     batch=16,              # Adjust based on your GPU VRAM
#     patience=20,           # Stop early if no improvement
    
#     # Aerial Optimization Flags
#     mosaic=1.0,            # 100% mosaic: Forces model to learn context
#     mixup=0.1,             # 10% mixup: Helps with crowded/occluded targets
#     copy_paste=0.3,        # 30% copy-paste: Great for small object instance learning
#     degrees=0.0,           # Disable rotation (drones usually stay level)
#     name="yolov8n_p2_visdrone"
# )

results = model.train(data="VisDrone.yaml")  # Standard VisDrone dataset config