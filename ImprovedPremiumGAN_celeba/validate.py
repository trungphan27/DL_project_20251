"""
Validation Script for ImprovedPremiumGAN
Computes all metrics on validation dataset:
- PSNR (masked region)
- PSNR (full image)
- SSIM (full image)
- VGG Perceptual Loss
- MSE (masked region)
- MSE (unmasked region)
- FID Score

Usage:
    python validate.py --num_images 3000 --data_dir "F:/Reconstructing_face_under_mask/dataset/without_mask"
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import urllib.request

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import Generator
from loss import VGGLoss


# ===================== Face Detector and Masking =====================

class FaceDetector:
    """OpenCV DNN Face Detector for creating masks on-the-fly."""
    
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Download models if not exist
        if not os.path.exists(prototxt_path):
            print("Downloading deploy.prototxt...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                prototxt_path
            )
        
        if not os.path.exists(caffemodel_path):
            print("Downloading caffemodel...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                caffemodel_path
            )
        
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    
    def detect_and_mask(self, image):
        """
        Detect face and apply black mask to lower face region.
        
        Args:
            image: BGR numpy array
        
        Returns:
            masked_image: BGR numpy array with mask applied
            mask_binary: Binary mask where True = masked region
        """
        h, w = image.shape[:2]
        masked_image = image.copy()
        mask_binary = np.zeros((h, w), dtype=bool)
        
        # Prepare blob for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Find best face detection
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2)
        
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            face_height = y2 - y1
            
            # Mask lower 45% of face (from 55% to 100%)
            mask_start_y = y1 + int(face_height * 0.55)
            mask_end_y = y2
            mask_x1, mask_x2 = x1, x2
            
            # Clip to image boundaries
            mask_start_y = max(0, mask_start_y)
            mask_end_y = min(h, mask_end_y)
            mask_x1 = max(0, mask_x1)
            mask_x2 = min(w, mask_x2)
            
            # Apply mask
            masked_image[mask_start_y:mask_end_y, mask_x1:mask_x2] = 0
            mask_binary[mask_start_y:mask_end_y, mask_x1:mask_x2] = True
        else:
            # Fallback: CelebA images are face-centered
            mask_start_y = int(h * 0.55)
            masked_image[mask_start_y:, :] = 0
            mask_binary[mask_start_y:, :] = True
        
        return masked_image, mask_binary


# ===================== Dataset =====================

class ValidationDataset(Dataset):
    """
    Dataset for validation - loads only ground truth images and creates masks on-the-fly.
    """
    def __init__(self, gt_dir, file_list, img_size=256, face_detector=None):
        self.gt_dir = gt_dir
        self.file_list = file_list
        self.img_size = img_size
        self.face_detector = face_detector
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        gt_path = os.path.join(self.gt_dir, img_name)
        
        # Load and resize image
        image_bgr = cv2.imread(gt_path)
        if image_bgr is None:
            # Return next image if loading fails
            return self.__getitem__((idx + 1) % len(self))
        
        # Resize to 256x256
        image_bgr = cv2.resize(image_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        # Create masked version
        masked_bgr, mask_binary = self.face_detector.detect_and_mask(image_bgr)
        
        # Convert BGR to RGB PIL
        gt_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
        
        gt_pil = Image.fromarray(gt_rgb)
        masked_pil = Image.fromarray(masked_rgb)
        
        # Transform to tensor
        gt_tensor = self.transform(gt_pil)
        masked_tensor = self.transform(masked_pil)
        
        # Keep numpy versions for metric calculation
        gt_np = gt_rgb.astype(np.float32) / 255.0  # [0, 1] range
        
        return masked_tensor, gt_tensor, gt_np, mask_binary


# ===================== Metric Functions =====================

def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] range."""
    return (tensor * 0.5) + 0.5


def tensor_to_numpy(tensor):
    """Convert tensor (B, C, H, W) to numpy (B, H, W, C) in [0, 1] range."""
    tensor = denormalize(tensor)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().permute(0, 2, 3, 1).numpy()


def compute_psnr_masked(gen_np, gt_np, mask):
    """Compute PSNR only in masked region."""
    if mask.sum() == 0:
        return 0.0
    
    gen_masked = gen_np[mask]
    gt_masked = gt_np[mask]
    
    mse = np.mean((gen_masked - gt_masked) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10(1.0 / mse)


def compute_psnr_full(gen_np, gt_np):
    """Compute PSNR on full image."""
    return psnr(gt_np, gen_np, data_range=1.0)


def compute_ssim_full(gen_np, gt_np):
    """Compute SSIM on full image."""
    return ssim(gt_np, gen_np, data_range=1.0, channel_axis=2, win_size=7)


def compute_mse_masked(gen_np, gt_np, mask):
    """Compute MSE only in masked region."""
    if mask.sum() == 0:
        return 0.0
    
    gen_masked = gen_np[mask]
    gt_masked = gt_np[mask]
    
    return np.mean((gen_masked - gt_masked) ** 2)


def compute_mse_unmasked(gen_np, gt_np, mask):
    """Compute MSE only in unmasked region."""
    unmasked = ~mask
    if unmasked.sum() == 0:
        return 0.0
    
    gen_unmasked = gen_np[unmasked]
    gt_unmasked = gt_np[unmasked]
    
    return np.mean((gen_unmasked - gt_unmasked) ** 2)


# ===================== FID Calculator =====================

class FIDCalculator:
    """Compute FID Score using InceptionV3 features."""
    
    def __init__(self, device):
        from torchvision.models import inception_v3, Inception_V3_Weights
        
        self.device = device
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = nn.Identity()
        self.model.eval()
        self.model.to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_features(self, images):
        """Extract features from images (B, C, H, W) in [-1, 1] range."""
        images = nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = denormalize(images)
        
        with torch.no_grad():
            features = self.model(images)
        
        return features.cpu().numpy()
    
    def compute_statistics(self, features):
        """Compute mean and covariance of features."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def compute_fid(self, real_features, fake_features):
        """Compute FID between real and fake feature distributions."""
        from scipy import linalg
        
        mu1, sigma1 = self.compute_statistics(real_features)
        mu2, sigma2 = self.compute_statistics(fake_features)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(fid)


# ===================== Model Loading =====================

def load_generator(checkpoint_dir, device):
    """Load the latest generator checkpoint."""
    gen = Generator(in_channels=3).to(device)
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                   if f.startswith('gen_') and f.endswith('.pth.tar')]
    
    if not checkpoints:
        raise FileNotFoundError(f"No generator checkpoints found in {checkpoint_dir}")
    
    latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    cp_path = os.path.join(checkpoint_dir, latest_cp)
    
    print(f"Loading generator from: {latest_cp}")
    checkpoint = torch.load(cp_path, map_location=device)
    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    
    return gen


# ===================== Main Validation =====================

def validate(args):
    """Main validation function."""
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Get file list (first N images)
    all_files = [f for f in os.listdir(args.data_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files = sorted(all_files)[:args.num_images]
    
    print(f"Found {len(all_files)} images in {args.data_dir}")
    print(f"Validating on first {len(all_files)} images (resized to 256x256)")
    
    # Initialize face detector
    print("Initializing face detector...")
    face_detector = FaceDetector()
    
    # Load generator model
    gen = load_generator(config.CHECKPOINT_DIR, device)
    
    # VGG Loss
    vgg_loss_fn = VGGLoss().to(device)
    
    # Dataset & DataLoader
    dataset = ValidationDataset(args.data_dir, all_files, config.IMG_SIZE, face_detector)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                        num_workers=0, pin_memory=True)  # num_workers=0 for face detector
    
    # Metrics accumulators
    psnr_masked_list = []
    psnr_full_list = []
    ssim_full_list = []
    vgg_loss_list = []
    mse_masked_list = []
    mse_unmasked_list = []
    
    # For FID
    real_features_list = []
    fake_features_list = []
    
    if args.compute_fid:
        print("Initializing FID calculator...")
        fid_calculator = FIDCalculator(device)
    
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS:")
    print("=" * 50)
    
    # Main validation loop
    with torch.no_grad():
        loop = tqdm(loader, desc="Inference", leave=True)
        
        for masked_tensor, gt_tensor, gt_np_batch, mask_batch in loop:
            masked_tensor = masked_tensor.to(device)
            gt_tensor = gt_tensor.to(device)
            
            # Generate reconstructed images
            gen_tensor = gen(masked_tensor)
            
            # Compute VGG Loss
            vgg_loss = vgg_loss_fn(gen_tensor, gt_tensor).item()
            vgg_loss_list.append(vgg_loss)
            
            # Convert to numpy for other metrics
            gen_np_batch_out = tensor_to_numpy(gen_tensor)
            
            # Per-image metrics
            batch_size = masked_tensor.size(0)
            for i in range(batch_size):
                gen_np = gen_np_batch_out[i]
                gt_np = gt_np_batch[i].numpy()
                mask = mask_batch[i].numpy()
                
                # Compute metrics
                psnr_masked = compute_psnr_masked(gen_np, gt_np, mask)
                psnr_full = compute_psnr_full(gen_np, gt_np)
                ssim_val = compute_ssim_full(gen_np, gt_np)
                mse_masked = compute_mse_masked(gen_np, gt_np, mask)
                mse_unmasked = compute_mse_unmasked(gen_np, gt_np, mask)
                
                psnr_masked_list.append(psnr_masked)
                psnr_full_list.append(psnr_full)
                ssim_full_list.append(ssim_val)
                mse_masked_list.append(mse_masked)
                mse_unmasked_list.append(mse_unmasked)
            
            # FID features
            if args.compute_fid:
                real_feat = fid_calculator.get_features(gt_tensor)
                fake_feat = fid_calculator.get_features(gen_tensor)
                real_features_list.append(real_feat)
                fake_features_list.append(fake_feat)
            
            # Update progress bar
            loop.set_postfix(
                PSNR_m=f"{np.mean(psnr_masked_list):.2f}",
                PSNR_f=f"{np.mean(psnr_full_list):.2f}",
                SSIM=f"{np.mean(ssim_full_list):.4f}"
            )
    
    # Aggregate results
    results = {
        "PSNR (masked region)": np.mean(psnr_masked_list),
        "PSNR (full image)": np.mean(psnr_full_list),
        "SSIM (full image)": np.mean(ssim_full_list),
        "VGG Perceptual Loss": np.mean(vgg_loss_list),
        "MSE (masked region)": np.mean(mse_masked_list),
        "MSE (unmasked region)": np.mean(mse_unmasked_list),
    }
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS:")
    print("=" * 50)
    for metric, value in results.items():
        if "PSNR" in metric:
            print(f"  {metric}:\t{value:.2f} dB")
        elif "MSE" in metric or "Loss" in metric:
            print(f"  {metric}:\t{value:.6f}")
        else:
            print(f"  {metric}:\t{value:.4f}")
    
    # Compute FID
    if args.compute_fid:
        print("\n🔄 Computing FID...")
        real_features = np.concatenate(real_features_list, axis=0)
        fake_features = np.concatenate(fake_features_list, axis=0)
        
        fid_score = fid_calculator.compute_fid(real_features, fake_features)
        results["FID Score"] = fid_score
        print(f"  FID Score:\t\t{fid_score:.2f}")
    
    print("=" * 50)
    
    # Save results to file
    result_path = os.path.join(config.RESULT_DIR, "validation_results.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of images: {len(all_files)}\n")
        f.write(f"Data directory: {args.data_dir}\n")
        f.write(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}\n\n")
        
        for metric, value in results.items():
            if "PSNR" in metric:
                f.write(f"{metric}: {value:.2f} dB\n")
            elif "MSE" in metric or "Loss" in metric:
                f.write(f"{metric}: {value:.6f}\n")
            else:
                f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\n✅ Results saved to: {result_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate GAN model on test images")
    parser.add_argument("--num_images", type=int, default=3000,
                        help="Number of images to validate on (first N images)")
    parser.add_argument("--data_dir", type=str, 
                        default=r"F:/Reconstructing_face_under_mask/dataset/without_mask",
                        help="Path to without_mask directory containing ground truth images")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--compute_fid", action="store_true", default=True,
                        help="Compute FID score (requires more memory)")
    
    args = parser.parse_args()
    
    validate(args)
