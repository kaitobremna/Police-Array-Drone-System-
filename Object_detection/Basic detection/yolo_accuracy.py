from ultralytics import YOLO

# 1. Load your pre-trained model
model = YOLO('yolov8n.pt') 

# 2. Run validation on the COCO-128 subset
print("Starting validation on COCO-128 subset...")

# save_plots=True will automatically create PR_curve.png
metrics = model.val(data='coco128.yaml', plots=True)

# 3. Find your results
# The plot is saved in the 'runs/detect/val' folder
print("\nValidation complete.")
print(f"Results, including PR_curve.png, are saved in: {metrics.save_dir}")