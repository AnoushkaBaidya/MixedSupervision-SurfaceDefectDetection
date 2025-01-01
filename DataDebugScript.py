import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config import Config  # Import the Config class
from data.input_yourdataset import YourDataset

# Instantiate the Config object
cfg = Config()

# Set required configurations for your dataset
cfg.DATASET = "YOURDATASET"  # Match the dataset name in your implementation
cfg.DATASET_PATH = "/Users/anoushka/Desktop/Projects/SurfaceDefectDL/MyTrainDataset"
cfg.NUM_SEGMENTED = 10  # Example value, set appropriately
cfg.TRAIN_NUM = 100  # Example value, set appropriately
cfg.INPUT_WIDTH = 256  # Set the input width of your model
cfg.INPUT_HEIGHT = 256  # Set the input height of your model
cfg.init_extra()  # Initialize extra parameters based on the dataset

# Initialize the dataset
kind = "TRAIN"  # Specify the type of dataset (TRAIN/VAL/TEST)
dataset = YourDataset(kind, cfg, root=cfg.DATASET_PATH)

# Print dataset statistics
print(f"Total samples: {dataset.len}")
print(f"Positive samples: {dataset.num_pos}")
print(f"Negative samples: {dataset.num_neg}")

# Visualize a sample
sample = dataset[0]  # Replace 0 with an index to test
print("Sample")
print(type(sample[0]))  # Should be <class 'torch.Tensor'>
print(sample[0].shape)  # Should match (3, INPUT_HEIGHT, INPUT_WIDTH)

for i in range(1):
        sample = dataset[i]
        if sample[0] is None or (len(sample) > 1 and sample[1] is None):
            print(f"Invalid sample at index {i}: {sample}")
            


