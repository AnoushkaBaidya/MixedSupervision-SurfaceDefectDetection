import os
import cv2
import torch
import numpy as np
from models import SegDecNet
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from evaluation import evaluate_decision  # Import evaluation function for performance metrics
import logging

# Setup logging for detailed monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Configuration
MODEL_PATH = "/home/ubuntu/Projects/MixedSupervision-SurfaceDefectDetection/RESULTS/TRAININGDATASET/experiment_1/models/final_state_dict.pth"  # Path to the saved model
TEST_IMAGES_DIR = "/home/ubuntu/Projects/TestImages/8_Nov_Test_Dataset"  # Directory containing test images
OUTPUT_DIR = "/home/ubuntu/Projects/MixedSupervision-SurfaceDefectDetection/SavePredictions"  # Directory to save predictions
EVAL_OUTPUT_DIR = "/home/ubuntu/Projects/MixedSupervision-SurfaceDefectDetection/EvaluationResults"  # Directory to save evaluation results
INPUT_WIDTH = 256  # Input image width for model processing
INPUT_HEIGHT = 256  # Input image height for model processing
INPUT_CHANNELS = 3  # Number of input channels (3 for RGB, 1 for grayscale)
CLASSIFICATION_THRESHOLD = 0.5  # Threshold for binary classification
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU

# Load Model
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the correct path is provided.")

logging.info("Loading the model weights from %s", MODEL_PATH)
model = SegDecNet(DEVICE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)  # Disable gradient multipliers as we are not training
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Load the model's state dictionary
model.eval()  # Set model to evaluation mode to disable dropout and batch normalization updates
logging.info("Model loaded and set to evaluation mode.")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
logging.info("Output directories ensured: %s and %s", OUTPUT_DIR, EVAL_OUTPUT_DIR)

# Initialize lists to store predictions and ground truth labels
predictions = []
ground_truth = []
image_names = []

# Process Images
logging.info("Starting image processing...")
for img_name in tqdm(os.listdir(TEST_IMAGES_DIR)):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)

    # Load and preprocess the image
    img = cv2.imread(img_path) if INPUT_CHANNELS == 3 else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        logging.warning("Unable to load image %s. Skipping.", img_name)
        continue

    try:
        img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    except Exception as e:
        logging.error("Error resizing image %s: %s", img_name, e)
        continue

    img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]
    img_t = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE) / 255.0

    with torch.no_grad():
        dec_out, seg_out = model(img_t)
        img_score = torch.sigmoid(dec_out).item()
        seg_pred = torch.sigmoid(seg_out).squeeze().cpu().numpy()

    label = 1 if "bad" in img_name.lower() else 0

    predictions.append(img_score)
    ground_truth.append(label)
    image_names.append(img_name)

    # Optional: save segmentation prediction
    # seg_save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_pred.png")
    # cv2.imwrite(seg_save_path, (seg_pred * 255).astype(np.uint8))

logging.info("Image processing completed.")

# Convert predictions and ground truth to numpy arrays
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Evaluate Metrics
logging.info("Evaluating model performance...")
metrics = evaluate_decision(
    run_dir=TEST_IMAGES_DIR,
    folds=np.zeros_like(ground_truth),
    ground_truth=ground_truth,
    img_names=image_names,
    predictions=predictions,
    prefix="",
    output_dir=EVAL_OUTPUT_DIR,
    save=True
)

logging.info("Evaluation completed. Results saved to %s", EVAL_OUTPUT_DIR)

print(f"Evaluation Results:\nAUROC: {metrics['AUC']:.4f}, Average Precision: {metrics['AP']:.4f}, Best F1 Score: {metrics['thresholds']['best']['F_measure']:.4f}")
logging.info("AUROC: %.4f, Average Precision: %.4f, Best F1 Score: %.4f", metrics['AUC'], metrics['AP'], metrics['thresholds']['best']['F_measure'])
