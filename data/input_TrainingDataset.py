import os
import torch
from PIL import Image
from data.dataset import Dataset
from torchvision.transforms import ToTensor, Resize


class TrainingDataset(Dataset):
    def __init__(self, kind, cfg, root):
        """
        Dataset for training models with images and masks.

        :param kind: 'TRAIN' or 'TEST', determines if the dataset is for training or testing.
        :param cfg: Configuration object containing input dimensions and other settings.
        :param root: Root directory of the dataset.
        """
        super().__init__(path=root, cfg=cfg, kind=kind)
        self.root = root
        self.kind = kind
        self.transform = ToTensor()
        self.resize = Resize((cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH))
        self.read_contents()

    def read_contents(self):
        """
        Reads dataset contents from the Positive, Negative, and Masks directories.
        """
        positive_dir = os.path.join(self.root, "Positive")
        negative_dir = os.path.join(self.root, "Negative")
        mask_dir = os.path.join(self.root, "Masks")

        # Check for existence of directories
        if not os.path.exists(positive_dir):
            raise FileNotFoundError(f"Positive directory not found: {positive_dir}")
        if not os.path.exists(negative_dir):
            raise FileNotFoundError(f"Negative directory not found: {negative_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.pos_samples = []
        self.neg_samples = []

        # Load positive samples with masks
        for img_name in sorted(os.listdir(positive_dir)):
            img_path = os.path.join(positive_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            # Ensure corresponding mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {img_name}, skipping.")
                continue

            image = self._load_image(img_path)
            mask = self._load_image(mask_path, is_mask=True)
            loss_mask = self._generate_loss_mask(mask)  # Generate a loss mask
            #self.pos_samples.append([image, mask, loss_mask, True, img_path, mask_path, img_name])
            self.pos_samples.append([image, mask, loss_mask, True, img_name])


        # Load negative samples without masks
        for img_name in sorted(os.listdir(negative_dir)):
            img_path = os.path.join(negative_dir, img_name)
            image = self._load_image(img_path)
            
            # Create a zero mask for negatives (indicating no defect)
            seg_mask = torch.zeros(1, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)
            loss_mask = torch.ones(1, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)  # Optional loss mask
            
            #self.neg_samples.append([image, seg_mask, loss_mask, False, img_path, None, img_name])
            self.neg_samples.append([image, seg_mask, loss_mask, False, img_name])

        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)
        self.len = len(self.pos_samples) + len(self.neg_samples)

        print(f"Loaded {self.num_pos} positive samples and {self.num_neg} negative samples.")
        self.init_extra()

    def __getitem__(self, index):
        """
        Retrieves a sample by index.
        """
        if index < len(self.pos_samples):
            return self.pos_samples[index]
        else:
            return self.neg_samples[index - len(self.pos_samples)]

    def _load_image(self, img_path, is_mask=False):
        """
        Loads and processes an image or mask.
        
        :param img_path: Path to the image file.
        :param is_mask: Whether the file is a mask (grayscale).
        :return: Transformed tensor of the image.
        """
        try:
            image = Image.open(img_path).convert("L" if is_mask else "RGB")
            image = self.resize(image)
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(1 if is_mask else 3, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)

    def _generate_loss_mask(self, mask):
        """
        Generates a loss mask using distance transform on the mask.

        :param mask: Tensor mask of the defect.
        :return: Loss mask tensor.
        """
        try:
            import numpy as np
            import cv2

            # Convert the mask tensor to a NumPy array
            np_mask = mask.squeeze().numpy()  # Remove channel dimension if it exists

            # Ensure the mask is binary (values 0 or 255)
            binary_mask = (np_mask > 0).astype(np.uint8) * 255  # Convert to 0 or 255

            # Apply distance transform
            loss_mask = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)

            # Normalize the loss mask to range [0, 1]
            loss_mask = loss_mask / loss_mask.max() if loss_mask.max() > 0 else loss_mask

            # Convert back to tensor and add channel dimension
            return torch.tensor(loss_mask, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"Error generating loss mask: {e}")
            return torch.zeros(1, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)
