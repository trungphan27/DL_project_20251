import os, random, cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def lower_face_mask(h, w):

    mask = np.ones((h, w), np.float32)

    top = int(0.50 * h)
    bottom = int(0.92 * h)
    left = int(0.20 * w)
    right = int(0.80 * w)

    mask[top:bottom, left:right] = 0

    kernel = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        kernel,
        ((left + right)//2, (top + bottom)//2),
        ((right-left)//2, (bottom-top)//2),
        0, 0, 360, 1, -1
    )

    mask = mask * kernel + (1 - kernel)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = (mask > 0.5).astype(np.float32)

    return torch.from_numpy(mask).unsqueeze(0)

def generate_mask(h, w):
    mask = np.ones((h, w), np.float32)

    y_jitter = random.randint(-6, 6)
    x_jitter = random.randint(-8, 8)

    top = int(0.60 * h) + y_jitter
    bottom = int(0.95 * h)
    left = int(0.22 * w) + x_jitter
    right = int(0.78 * w) + x_jitter

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, h)
    right = min(right, w)

    mask[top:bottom, left:right] = 0
    face_shape = np.ones((h, w), np.uint8)
    cv2.ellipse(
        face_shape,
        ((left + right)//2, (top + bottom)//2),
        (int((right-left)*0.45), int((bottom-top)*0.55)),
        0, 0, 360, 0, -1
    )

    mask = np.minimum(mask, face_shape)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = np.clip(mask, 0, 1)

    return torch.from_numpy(mask).unsqueeze(0)


class FaceDataset(Dataset):
    def __init__(self, img_dir, size=256):
        self.imgs = [
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        img = self.transform(img)

        mask = generate_mask(img.shape[1], img.shape[2])
        mask = mask.float()
        masked_img = img * mask

        return masked_img, mask, img

