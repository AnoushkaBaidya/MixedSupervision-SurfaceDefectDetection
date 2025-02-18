import sys
import os
import matplotlib.pyplot as plt
import torch
from config import Config  # Import the Config class
from data.input_OnDemand import TrainingDataset  # Import the on-demand dataset class

def visualize_sample(image, mask, loss_mask, label, img_name):
    """
    Visualizes a sample with its image, mask, and loss mask.
    """
    # Convert tensors to NumPy arrays for visualization
    if image.dim() == 3 and image.shape[0] in [1, 3]:
        image_np = image.numpy().transpose(1, 2, 0)  # CHW to HWC
    else:
        image_np = image.numpy()

    if mask is not None:
        mask_np = mask.squeeze().numpy()  # Squeeze for grayscale
    else:
        mask_np = None

    if loss_mask is not None:
        loss_mask_np = loss_mask.squeeze().numpy()
    else:
        loss_mask_np = None

    # Plot the sample
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title(f"Image: {img_name} (Label: {'Positive' if label == 1 else 'Negative'})")
    axes[0].axis("off")

    if mask_np is not None:
        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Corresponding Mask")
        axes[1].axis("off")
    else:
        axes[1].imshow([[0]], cmap="gray")  # Empty plot
        axes[1].set_title("No Mask (Negative Sample)")
        axes[1].axis("off")

    if loss_mask_np is not None:
        axes[2].imshow(loss_mask_np, cmap="jet")
        axes[2].set_title("Loss Mask (Generated)")
        axes[2].axis("off")
    else:
        axes[2].imshow([[0]], cmap="gray")  # Empty plot
        axes[2].set_title("No Loss Mask (Negative Sample)")
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()


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

    # Visualize all parts of dataset[0]
    print("Visualizing all components of the first dataset sample (dataset[0]):")
    try:
        image, mask, loss_mask, label, img_name = dataset[0]
        visualize_sample(image, mask, loss_mask, label.item(), img_name)
    except Exception as e:
        print(f"Error visualizing dataset[0]: {e}")

    # Iterate through the dataset and visualize random samples
    print("\nIterating through the dataset and displaying one positive and one negative sample...")

    # Display a positive sample
    if dataset.num_pos > 0:
        print("Displaying an example positive image, mask, and loss mask...")
        try:
            pos_image, pos_mask, pos_loss_mask, pos_label, pos_img_name = dataset[10]
            visualize_sample(pos_image, pos_mask, pos_loss_mask, pos_label.item(), pos_img_name)
        except Exception as e:
            print(f"Error visualizing positive sample: {e}")
    else:
        print("No positive samples found in the dataset.")

    # Display a negative sample
    if dataset.num_neg > 0:
        print("Displaying an example negative image and default loss mask...")
        try:
            neg_image, neg_mask, neg_loss_mask, neg_label, neg_img_name = dataset[dataset.num_pos + 10]
            visualize_sample(neg_image, neg_mask, neg_loss_mask, neg_label.item(), neg_img_name)
        except Exception as e:
            print(f"Error visualizing negative sample: {e}")
    else:
        print("No negative samples found in the dataset.")
