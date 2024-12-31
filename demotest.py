from models import SegDecNet
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/RESULTS/TRAININGDATASET/experiment_1/models/best_state_dict.pth"
TEST_IMAGE_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/TestImages/Code03117.png"
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
INPUT_CHANNELS = 3
DEVICE = "cpu"  # Change to "cuda" if using GPU

# Classification threshold
CLASSIFICATION_THRESHOLD = 0.5

# Load Model
model = SegDecNet(DEVICE, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Set model to evaluation mode

# Load and Preprocess Image
img = cv2.imread(TEST_IMAGE_PATH) if INPUT_CHANNELS == 3 else cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]  # [C x H x W]
img_t = torch.from_numpy(img).unsqueeze(0).float().to(DEVICE) / 255.0  # [1 x C x H x W]

# Inference
with torch.no_grad():  # Disable gradient tracking
    dec_out, seg_out = model(img_t)
    img_score = torch.sigmoid(dec_out).item()  # Classification score
    seg_pred = torch.sigmoid(seg_out).detach().squeeze().cpu().numpy()  # Detach before converting to NumPy

# Determine if it's a good sample or not
if img_score > CLASSIFICATION_THRESHOLD:
    sample_status = "NOT a Good Sample (Defective)"
else:
    sample_status = "Good Sample"

# Print Classification Score and Sample Status
print(f"Classification Score: {img_score:.4f}")
print(f"Sample Status: {sample_status}")

# Visualize Results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title(f"Input Image\n{sample_status}")

# Properly reshape for RGB or grayscale
if INPUT_CHANNELS == 3:
    plt.imshow(img.transpose(1, 2, 0))  # [C, H, W] -> [H, W, C]
else:
    plt.imshow(img[0], cmap="gray")  # Grayscale image

plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(seg_pred, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
