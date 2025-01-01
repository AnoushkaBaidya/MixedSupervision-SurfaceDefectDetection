import sys
import matplotlib.pyplot as plt
from config import Config
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config import Config  # Import the Config class
from data.input_TrainingDataset import TrainingDataset 



if __name__ == "__main__":
    # Initialize configuration
    cfg = Config()
    cfg.INPUT_HEIGHT = 256  # Example height
    cfg.INPUT_WIDTH = 256   # Example width
    cfg.DATASET_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/MyTrainDataset"  # Updated with the dataset path

    # Initialize the dataset
    kind = "TRAIN"
    dataset = TrainingDataset(kind, cfg, root=cfg.DATASET_PATH)

    # Print dataset details
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of positive samples: {dataset.num_pos}")
    print(f"Number of negative samples: {dataset.num_neg}")

    # Display a positive sample and its mask
    if dataset.num_pos > 0:
        print("Displaying an example positive image and its corresponding mask...")

        # Get the first positive sample
        example_sample = dataset.pos_samples[3]
        #image, mask, loss_mask, is_segmented, img_path, mask_path, sample_name = example_sample
        image, mask, loss_mask, is_segmented, sample_name = example_sample


        # Convert tensors to NumPy arrays for visualization
        image_np = image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        mask_np = mask.squeeze().numpy()  # Remove channel dimension for grayscale mask

        # Plot the image and its mask
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Positive Image: {sample_name}")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Corresponding Mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("No positive samples found in the dataset.")
