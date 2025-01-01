import os
import cv2
import torch
import numpy as np
from models import SegDecNet
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from evaluation import evaluate_decision  # Import evaluation function for performance metrics

# Configuration
MODEL_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/RESULTS/TRAININGDATASET/experiment_1/models/best_state_dict.pth"  # Path to the saved model
TEST_IMAGES_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8_Nov_Test_Dataset"  # Directory containing test images
OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/SavePredictions"  # Directory to save predictions
EVAL_OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/EvaluationResults"  # Directory to save evaluation results
INPUT_WIDTH = 256  # Input image width for model processing
INPUT_HEIGHT = 256  # Input image height for model processing
INPUT_CHANNELS = 3  # Number of input channels (3 for RGB, 1 for grayscale)
CLASSIFICATION_THRESHOLD = 0.5  # Threshold for binary classification
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU

# Load Model
# Initialize the model and set it to evaluation mode
model = SegDecNet(DEVICE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)  # Disable gradient multipliers as we are not training
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Load the model's state dictionary
model.eval()  # Set model to evaluation mode to disable dropout and batch normalization updates

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory for predictions if it doesn't exist
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)  # Create directory for evaluation results if it doesn't exist

# Initialize lists to store predictions and ground truth labels
predictions = []  # Store predicted classification scores
ground_truth = []  # Store ground truth labels
image_names = []  # Store image filenames

# Process Images
# Iterate through all images in the test directory
for img_name in tqdm(os.listdir(TEST_IMAGES_DIR)):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)  # Full path to the image

    # Load and preprocess the image
    img = cv2.imread(img_path) if INPUT_CHANNELS == 3 else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Warning: Unable to load image {img_name}. Skipping.")
        continue

    try:
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))  # Resize image to match model input dimensions
    except Exception as e:
        print(f"Error resizing image {img_name}: {e}")
        continue

    # Preprocess image for the model
    img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]  # Rearrange dimensions to match model input
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE) / 255.0  # Normalize and convert to tensor

    with torch.no_grad():  # Disable gradient computation for inference
        dec_out, seg_out = model(img_t)  # Get classification and segmentation outputs
        img_score = torch.sigmoid(dec_out).item()  # Compute classification score using sigmoid
        seg_pred = torch.sigmoid(seg_out).squeeze().cpu().numpy()  # Predicted segmentation mask

    # Simulate ground truth label based on file naming convention (e.g., "bad" indicates defect)
    label = 1 if "bad" in img_name.lower() else 0

    # Save predictions and ground truth for evaluation
    predictions.append(img_score)  # Append predicted score
    ground_truth.append(label)  # Append ground truth label
    image_names.append(img_name)  # Append image name

    # Save segmentation prediction (optional - uncomment if needed)
    # seg_save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_pred.png")
    # cv2.imwrite(seg_save_path, (seg_pred * 255).astype(np.uint8))

# Convert predictions and ground truth to numpy arrays
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Evaluate Metrics
# Use the evaluation function to calculate performance metrics
metrics = evaluate_decision(
    run_dir=TEST_IMAGES_DIR,  # Directory of test images
    folds=np.zeros_like(ground_truth),  # Assuming single fold for simplicity
    ground_truth=ground_truth,  # Ground truth labels
    img_names=image_names,  # Image filenames
    predictions=predictions,  # Predicted scores
    prefix="",  # Prefix for saving evaluation results
    output_dir=EVAL_OUTPUT_DIR,  # Output directory for evaluation results
    save=True  # Save evaluation results to disk
)

# Print Evaluation Results
# Display key metrics including AUROC, Average Precision, and Best F1 Score
print(f"Evaluation Results:\nAUROC: {metrics['AUC']:.4f}, Average Precision: {metrics['AP']:.4f}, Best F1 Score: {metrics['thresholds']['best']['F_measure']:.4f}")
