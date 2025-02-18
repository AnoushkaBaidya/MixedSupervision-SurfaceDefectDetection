import sys
import os
import matplotlib.pyplot as plt
import torch
from config import Config  # Import the Config class
from data.input_OnDemand import TrainingDataset  # Import the on-demand dataset class

if __name__ == "__main__":
    # Initialize configuration
    cfg = Config()
    cfg.INPUT_HEIGHT = 256  # Example height
    cfg.INPUT_WIDTH = 256   # Example width
    cfg.DATASET_PATH = "/Users/anoushka/Desktop/Projects/MixedSupervision-SurfaceDefectDetection/MyTrainDataset"  # Update with the dataset path

    # Initialize the dataset
    kind = "TRAIN"
    dataset = TrainingDataset(kind, cfg, root=cfg.DATASET_PATH)

    # Print dataset details
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of positive samples: {dataset.num_pos}")
    print(f"Number of negative samples: {dataset.num_neg}")

    # Iterate through the dataset and check on-demand loading
    for idx in range(len(dataset)):
        try:
            if idx < dataset.num_pos:
                img_path, mask_path, img_name = dataset.pos_samples[idx]
                mask = dataset._load_image(mask_path, is_mask=True)
            else:
                img_path, img_name = dataset.neg_samples[idx - dataset.num_pos]
                mask = None

            image = dataset._load_image(img_path)
        except Exception as e:
            print(f"Failed to load image at index {idx}: {e}")

    # Display a positive sample and its mask
    if dataset.num_pos > 0:
        print("Displaying an example positive image and its corresponding mask...")

        try:
            # Get the first positive sample
            img_path, mask_path, img_name = dataset.pos_samples[10]
            image = dataset._load_image(img_path)
            mask = dataset._load_image(mask_path, is_mask=True)

            # Convert tensors to NumPy arrays for visualization
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image_np = image.numpy().transpose(1, 2, 0)  # CHW to HWC
            else:
                image_np = image.numpy()

            mask_np = mask.squeeze().numpy()  # Squeeze for grayscale

            # Plot the image and its mask
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image_np)
            axes[0].set_title(f"Positive Image: {img_name}")
            axes[0].axis("off")

            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("Corresponding Mask")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error visualizing positive sample: {e}")
    else:
        print("No positive samples found in the dataset.")

    # Display a negative sample
    if dataset.num_neg > 0:
        print("Displaying an example negative image...")

        try:
            # Get the first negative sample
            img_path, img_name = dataset.neg_samples[10]
            image = dataset._load_image(img_path)

            # Convert tensor to NumPy array for visualization
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                image_np = image.numpy().transpose(1, 2, 0)  # CHW to HWC
            else:
                image_np = image.numpy()

            # Plot the image
            plt.figure(figsize=(5, 5))
            plt.imshow(image_np)
            plt.title(f"Negative Image: {img_name}")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print(f"Error visualizing negative sample: {e}")
    else:
        print("No negative samples found in the dataset.")
