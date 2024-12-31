import os
import cv2
import torch
import numpy as np
from models import SegDecNet
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from evaluation import evaluate_decision  # Import evaluation function

# Configuration
MODEL_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/RESULTS/TRAININGDATASET/experiment_1/models/best_state_dict.pth"
TEST_IMAGES_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8_Nov_Test_Dataset"
OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/SavePredictions"
EVAL_OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/EvaluationResults"  # Directory for evaluation results
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INPUT_CHANNELS = 3
CLASSIFICATION_THRESHOLD = 0.5  # Classification threshold
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model
model = SegDecNet(DEVICE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Set model to evaluation mode

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# Initialize lists to store predictions and ground truth labels
predictions = []
ground_truth = []
image_names = []

# Process Images
for img_name in tqdm(os.listdir(TEST_IMAGES_DIR)):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)

    # Load and preprocess the image
    img = cv2.imread(img_path) if INPUT_CHANNELS == 3 else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if img is None:
        print(f"Warning: Unable to load image {img_name}. Skipping.")
        continue

    try:
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    except Exception as e:
        print(f"Error resizing image {img_name}: {e}")
        continue

    img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE) / 255.0

    with torch.no_grad():
        dec_out, seg_out = model(img_t)  # Get classification and segmentation outputs
        img_score = torch.sigmoid(dec_out).item()  # Classification score
        seg_pred = torch.sigmoid(seg_out).squeeze().cpu().numpy()  # Predicted segmentation mask

    # Simulate ground truth based on file name (e.g., "good" -> 0, "bad" -> 1)
    label = 1 if "bad" in img_name.lower() else 0

    # Save predictions and ground truth
    predictions.append(img_score)
    ground_truth.append(label)
    image_names.append(img_name)

    # Save segmentation prediction (optional)
    seg_save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_pred.png")
    cv2.imwrite(seg_save_path, (seg_pred * 255).astype(np.uint8))

# Convert to numpy arrays
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Evaluate Metrics
metrics = evaluate_decision(
    run_dir=TEST_IMAGES_DIR,
    folds=np.zeros_like(ground_truth),  # Assuming single fold for simplicity
    ground_truth=ground_truth,
    img_names=image_names,
    predictions=predictions,
    prefix="",
    output_dir=EVAL_OUTPUT_DIR,
    save=True
)

# Print Evaluation Results
print(f"Evaluation Results:\nAUROC: {metrics['AUC']:.4f}, Average Precision: {metrics['AP']:.4f}, Best F1 Score: {metrics['thresholds']['best']['F_measure']:.4f}")
