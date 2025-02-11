import sys
import os
import matplotlib.pyplot as plt
import torch
from config import Config  # Import the Config class
from data.input_OnDemand import TrainingDatasetOnDemand  # Import the on-demand dataset class

if __name__ == "__main__":
    # Initialize configuration
    cfg = Config()
    cfg.INPUT_HEIGHT = 256  # Example height
    cfg.INPUT_WIDTH = 256   # Example width
    cfg.DATASET_PATH = "/Users/anoushka/Desktop/Projects/MixedSupervision-SurfaceDefectDetection/MyTrainDataset"  # Update with the dataset path

    # Initialize the dataset
    kind = "TRAIN"
    dataset = TrainingDatasetOnDemand(kind, cfg, root=cfg.DATASET_PATH)

    # Print dataset details
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of positive samples: {dataset.num_pos}")
    print(f"Number of negative samples: {dataset.num_neg}")

    # Iterate through the dataset and check on-demand loading
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]  # Access dynamically loaded data
            img_path = sample[4]  # Image path
            mask_path = sample[5] if sample[3] else None  # Mask path if positive

            # Load the image and mask on demand
            image = dataset._load_image(img_path)
            if mask_path:
                mask = dataset._load_image(mask_path, is_mask=True)
        except Exception as e:
            print(f"Failed to load image at index {idx}: {e}")

    # Display a positive sample and its mask
    if dataset.num_pos > 0:
        print("Displaying an example positive image and its corresponding mask...")

        # Get the first positive sample
        example_sample = dataset[0]  # Accessing dynamically loaded data
        image, mask, loss_mask, is_segmented, img_path, mask_path, sample_name = example_sample

        # Load the image on demand
        image = dataset._load_image(img_path)
        mask = dataset._load_image(mask_path, is_mask=True) if mask_path else torch.zeros(dataset.image_size)

        if image.dim() == 4:  # If batch dimension exists, remove it
            image = image.squeeze(0)
        # Convert tensors to NumPy arrays for visualization
        image_np = image.numpy().transpose(1, 2, 0) if image.shape[0] in [1, 3] else image.numpy()  # Ensure correct shape
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

    # Display a negative sample
    if dataset.num_neg > 0:
        print("Displaying an example negative image...")

        # Get the first negative sample
        example_sample = dataset[dataset.num_pos]  # Accessing dynamically loaded data
        image, mask, loss_mask, is_segmented, img_path, mask_path, sample_name = example_sample

        # Load the image on demand
        image = dataset._load_image(img_path)

        if image.dim() == 4:  # If batch dimension exists, remove it
            image = image.squeeze(0)

        # Convert tensors to NumPy arrays for visualization
        image_np = image.numpy().transpose(1, 2, 0) if image.shape[0] in [1, 3] else image.numpy()  # Ensure correct shape

        # Plot the image
        plt.figure(figsize=(5, 5))
        plt.imshow(image_np)
        plt.title(f"Negative Image: {sample_name}")
        plt.axis("off")
        plt.show()
    else:
        print("No negative samples found in the dataset.")