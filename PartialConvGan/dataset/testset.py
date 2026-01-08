import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class Testset(Dataset):
    def __init__(self, npy_path):
        self.images = np.load(npy_path)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], "RGB")
        image = self.transform(image)

        size = 256
        mask = np.ones((size, size), dtype=np.float32)
        mask_top = int(size * 0.55)
        mask_bottom = int(size * 0.95)
        mask_left = int(size * 0.20)
        mask_right = int(size * 0.80)
        mask[mask_top:mask_bottom, mask_left:mask_right] = 0.0

        mask = torch.from_numpy(mask).unsqueeze(0)
        masked_image = image * mask

        return masked_image, mask, image
