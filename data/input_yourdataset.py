import os
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import torch
from data.dataset import Dataset

class YourDataset(Dataset):
    def __init__(self, kind, cfg, root):
        super().__init__(path=root, cfg=cfg, kind=kind)
        print("cfg type:", type(cfg))
        self.root = root
        self.transform = ToTensor()  # Add transformations for images and masks
        self.resize = Resize((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH))  # Resize images and masks to input dimensions
        self.read_contents()

    def read_contents(self):
        good_dir = os.path.join(self.root, "good")
        bad_dir = os.path.join(self.root, "bad")
        mask_dir = os.path.join(self.root, "masks")

        if not os.path.exists(good_dir):
            print(f"Good directory not found: {good_dir}")
        if not os.path.exists(bad_dir):
            print(f"Bad directory not found: {bad_dir}")
        if not os.path.exists(mask_dir):
            print(f"Mask directory not found: {mask_dir}")

        self.pos_samples = []
        self.neg_samples = []

        # Placeholder tensor for missing masks
        placeholder_mask = torch.zeros(1, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)

        # Load positive (good) samples
        for img_name in os.listdir(good_dir):
            img_path = os.path.join(good_dir, img_name)
            image = self._load_image(img_path)
            self.pos_samples.append([image, placeholder_mask, placeholder_mask, False, img_name])

        # Load negative (bad) samples and masks
        for img_name in os.listdir(bad_dir):
            img_path = os.path.join(bad_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            # Check if mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {img_path}, skipping.")
                continue  # Skip this sample if mask is missing
            image = self._load_image(img_path)
            mask = self._load_image(mask_path, is_mask=True)
            self.neg_samples.append([image, mask, mask, True, img_name])

        for idx, sample in enumerate(self.neg_samples):
            if sample[0] is None or sample[1] is None:
                print(f"Bad sample at index {idx}: Image or mask is None")    

        self.len = len(self.pos_samples) + len(self.neg_samples)
        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)
        print(f"Loaded {self.num_pos} good and {self.num_neg} bad samples.")

        self.init_extra()

    def __getitem__(self, index):
        if index < len(self.pos_samples):
            sample = self.pos_samples[index]
        else:
            sample = self.neg_samples[index - len(self.pos_samples)]
    
        if sample is None:
            raise ValueError(f"Sample at index {index} is None. Check your data loading process.")
        return sample

    def _load_image(self, img_path, is_mask=False):
        try:
            image = Image.open(img_path).convert("L" if is_mask else "RGB")
            image = self.resize(image)  # Resize image
            image = self.transform(image)  # Convert to tensor
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(1 if is_mask else 3, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)


"""
# Debug script example:
if __name__ == "__main__":
    import sys
    from config import Config

    # Initialize configuration
    cfg = Config()
    cfg.INPUT_HEIGHT = 256  # Example height
    cfg.INPUT_WIDTH = 256   # Example width
    cfg.DATASET_PATH = "/path/to/your/dataset"  # Update with your dataset path

    kind = "TRAIN"
    dataset = YourDataset(kind, cfg, root=cfg.DATASET_PATH)

    # Test dataset
    print(f"Dataset length: {len(dataset)}")
    for i in range(5):  # Print first 5 samples
        sample = dataset[i]
        print(f"Sample {i}: {sample}")
"""