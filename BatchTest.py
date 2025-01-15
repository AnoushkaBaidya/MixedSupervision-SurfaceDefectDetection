""" CALCULATES WITH OUR F1SCORE AND AUROC SCORE"""

import os
import cv2
import torch
import numpy as np
from models import SegDecNet
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

# Configuration
MODEL_PATH = "/Users/anoushka/Desktop/Projects/MixedSupervision-SurfaceDefectDetection/RESULTS/TRAININGDATASET/experiment_1/models/best_state_dict.pth"  # Path to the saved model
TEST_IMAGES_DIR = "/Users/anoushka/Desktop/Projects/MixedSupervision-SurfaceDefectDetection/TestImages/8_Nov_Test_Dataset"  # Directory containing test images
OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/MixedSupervision-SurfaceDefectDetection//SavePredictions"  # Directory to save predictions
EVAL_OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/MixedSupervision-SurfaceDefectDetection/EvaluationResults"  # Directory to save evaluation results
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INPUT_CHANNELS = 3
CLASSIFICATION_THRESHOLD = 0.5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model
model = SegDecNet(DEVICE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lists to store predictions and ground truth labels
predictions = []
ground_truth = []
image_names = []

# Process Images
for img_name in tqdm(os.listdir(TEST_IMAGES_DIR)):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    img = cv2.imread(img_path) if INPUT_CHANNELS == 3 else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

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
        dec_out, seg_out = model(img_t)
        img_score = torch.sigmoid(dec_out).item()
        seg_pred = torch.sigmoid(seg_out).squeeze().cpu().numpy()

    # Simulate ground truth label based on file naming convention
    label = 1 if "bad" in img_name.lower() else 0

    # Save predictions and ground truth for evaluation
    predictions.append(img_score)
    ground_truth.append(label)
    image_names.append(img_name)

    # Print classification result (optional)
    status = "Defective" if img_score > CLASSIFICATION_THRESHOLD else "Good"
    print(f"Image: {img_name}, Score: {img_score:.4f}, Status: {status}, Label: {'Bad' if label == 1 else 'Good'}")

# Convert predictions and ground truth to numpy arrays
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Calculate F1 score and AUROC
binary_predictions = (predictions > CLASSIFICATION_THRESHOLD).astype(int)
f1 = f1_score(ground_truth, binary_predictions)
auroc = roc_auc_score(ground_truth, predictions)

print(f"\nF1 Score: {f1:.4f}")
print(f"AUROC Score: {auroc:.4f}")
