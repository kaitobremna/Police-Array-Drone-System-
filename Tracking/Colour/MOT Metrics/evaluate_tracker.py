import motmetrics as mm
import pandas as pd
import os

# ==========================================
# 1. FILE PATHS (Update these to your local paths)
# ==========================================
# The human-annotated perfect tracking data
ground_truth_file = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/datasets/VisDrone2019-MOT-test-dev/annotations/uav0000073_00600_v.txt"

# The text file your Ultralytics tracking script just generated
tracking_results_file = "/Users/kaitobremner/Desktop/Studies/Y3/3YP/Drone_software/Police-Array-Drone-System-/runs/YOLO26l_tracks/uav0000073_00600_v_mot/results.txt"

# ==========================================
# 2. DEFINE THE EVALUATOR
# ==========================================
def evaluate_tracking(gt_path, ts_path):
    print(f"📊 Loading Ground Truth: {gt_path}")
    print(f"🤖 Loading Tracker Results: {ts_path}\n")
    
    # Create a MOT accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Load the text files using standard MOT15 2D format
    # format: frame, id, x, y, w, h, conf, -1, -1, -1
    gt = mm.io.loadtxt(gt_path, fmt='mot15-2D', min_confidence=1)
    ts = mm.io.loadtxt(ts_path, fmt='mot15-2D')

    # Compare ground truth to tracker using Intersection over Union (IoU)
    # distth=0.5 means boxes must overlap by at least 50% to be considered a match
    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    
    # Define the specific metrics we want to see (The ones you cited in your thesis!)
    metrics_list = [
        'num_frames',        # Total frames
        'idf1',              # Identification F1-Score (Your most important metric)
        'mota',              # Multi-Object Tracking Accuracy
        'motp',              # Multi-Object Tracking Precision
        'num_false_positives', # False Positives (FP)
        'num_misses',        # False Negatives (FN)
        'num_switches'       # Identity Switches (IDSW)
    ]

    # Calculate metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=metrics_list, name='YOLO26l_BoTSORT')
    
    # Render and print the summary cleanly
    str_summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap=mm.io.motchallenge_metric_names
    )
    
    print("==========================================================================")
    print("                      TRACKING EVALUATION RESULTS                         ")
    print("==========================================================================")
    print(str_summary)
    print("==========================================================================")

# ==========================================
# 3. RUN EVALUATION
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(ground_truth_file):
        print(f"❌ ERROR: Ground truth file not found at {ground_truth_file}")
    elif not os.path.exists(tracking_results_file):
        print(f"❌ ERROR: Tracking results file not found at {tracking_results_file}")
    else:
        evaluate_tracking(ground_truth_file, tracking_results_file)