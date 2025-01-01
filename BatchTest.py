import os
import cv2
import torch
import numpy as np
from models import SegDecNet
from tqdm import tqdm

# This code is for quick inference and classification to inspect model behavior and outputs,
# suitable for a lighter debugging or demonstration context.

# Configuration
MODEL_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/RESULTS/TRAININGDATASET/experiment_1/models/best_state_dict.pth"  # Path to the trained model
TEST_IMAGES_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/8_Nov_Test_Dataset"  # Directory containing test images
OUTPUT_DIR = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/SavePredictions"  # Directory to save segmentation predictions
INPUT_WIDTH = 256  # Input image width for resizing
INPUT_HEIGHT = 256  # Input image height for resizing
INPUT_CHANNELS = 3  # Number of input channels (3 for RGB, 1 for grayscale)
CLASSIFICATION_THRESHOLD = 0.5  # Threshold for binary classification
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use GPU if available, else CPU

# Load Model
# Initialize the model and set it to evaluation mode
model = SegDecNet(DEVICE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)  # Disable gradient multipliers for inference
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Load the trained model weights
model.eval()  # Set model to evaluation mode to disable dropout and batch normalization updates

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory for saving predictions if it doesn't exist

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

    # Determine the sample status based on the classification score
    sample_status = "NOT a Good Sample (Defective)" if img_score > CLASSIFICATION_THRESHOLD else "Good Sample"

    # Print classification result
    print(f"Image: {img_name}, Classification Score: {img_score:.4f}, Status: {sample_status}")

    # Save segmentation prediction (optional - uncomment if needed)
    # seg_save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_pred.png")
    # cv2.imwrite(seg_save_path, (seg_pred * 255).astype(np.uint8))
