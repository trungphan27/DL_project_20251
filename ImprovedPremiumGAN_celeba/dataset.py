
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import config


class CelebAMaskDataset(Dataset):
    """
    Dataset for CelebA-HQ with pre-generated masks.
    Loads paired images from without_mask/ (ground truth) and with_mask/ (masked input).
    """
    def __init__(self, file_list, gt_dir, masked_dir, transforms=None):
        """
        Args:
            file_list: List of image filenames
            gt_dir: Directory containing ground truth images (without_mask/)
            masked_dir: Directory containing masked images (with_mask/)
            transforms: Torchvision transforms to apply
        """
        self.file_list = file_list
        self.gt_dir = gt_dir
        self.masked_dir = masked_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        
        # Load ground truth image (without mask)
        gt_path = os.path.join(self.gt_dir, img_name)
        # Load masked image (with black rectangle mask)
        masked_path = os.path.join(self.masked_dir, img_name)
        
        try:
            gt_image = Image.open(gt_path).convert("RGB")
            masked_image = Image.open(masked_path).convert("RGB")
        except Exception as e:
            # Fallback to next image if loading fails
            print(f"Warning: Failed to load {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms:
            gt_image = self.transforms(gt_image)
            masked_image = self.transforms(masked_image)

        # Return (masked_input, ground_truth) for training
        return masked_image, gt_image


def get_transforms():
    """Get preprocessing transforms for CelebA-HQ 256x256 images."""
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])
