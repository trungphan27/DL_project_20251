"""
Test Script for ImprovedPremiumGAN Face Reconstruction
======================================================

Usage:
    python test.py --image path/to/face_image.jpg
    python test.py --image path/to/face_image.jpg --output results/

This script:
1. Loads a face image
2. Detects face using OpenCV DNN
3. Creates a black mask on the lower face region
4. Runs the Generator to reconstruct the masked region
5. Compares with ground truth and prints metrics
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
import argparse
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import urllib.request

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import Generator
from loss import VGGLoss


# ===================== Face Detector =====================

class FaceDetector:
    """OpenCV DNN Face Detector for creating masks."""
    
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        prototxt_path = os.path.join(model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Download if not exist
        if not os.path.exists(prototxt_path):
            print("Downloading face detector prototxt...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                prototxt_path
            )
        
        if not os.path.exists(caffemodel_path):
            print("Downloading face detector model...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                caffemodel_path
            )
        
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        print("✓ Face detector loaded")
    
    def detect_and_mask(self, image):
        """
        Detect face and apply black mask to lower face region.
        
        Returns:
            masked_image: Image with black mask on lower face
            mask_binary: Binary mask (True = masked region)
            face_box: Face bounding box (x1, y1, x2, y2) or None
        """
        h, w = image.shape[:2]
        masked_image = image.copy()
        mask_binary = np.zeros((h, w), dtype=bool)
        face_box = None
        
        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Find best detection
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
            face_box = best_box
            face_height = y2 - y1
            
            # Mask lower 45% of face (from 55% height)
            mask_start_y = max(0, y1 + int(face_height * 0.55))
            mask_end_y = min(h, y2)
            mask_x1, mask_x2 = max(0, x1), min(w, x2)
            
            masked_image[mask_start_y:mask_end_y, mask_x1:mask_x2] = 0
            mask_binary[mask_start_y:mask_end_y, mask_x1:mask_x2] = True
        else:
            # Fallback for face-centered images
            print("⚠ No face detected, using fallback mask")
            mask_start_y = int(h * 0.55)
            masked_image[mask_start_y:, :] = 0
            mask_binary[mask_start_y:, :] = True
        
        return masked_image, mask_binary, face_box


# ===================== Metrics =====================

def compute_metrics(gen_np, gt_np, mask):
    """Compute all metrics between generated and ground truth."""
    
    # PSNR (masked region)
    if mask.sum() > 0:
        mse_masked = np.mean((gen_np[mask] - gt_np[mask]) ** 2)
        psnr_masked = 10 * np.log10(1.0 / mse_masked) if mse_masked > 0 else float('inf')
    else:
        psnr_masked = 0.0
        mse_masked = 0.0
    
    # PSNR (full image)
    psnr_full = psnr(gt_np, gen_np, data_range=1.0)
    
    # SSIM (full image)
    ssim_full = ssim(gt_np, gen_np, data_range=1.0, channel_axis=2, win_size=7)
    
    # MSE (masked region)
    mse_masked = np.mean((gen_np[mask] - gt_np[mask]) ** 2) if mask.sum() > 0 else 0.0
    
    # MSE (unmasked region)
    unmasked = ~mask
    mse_unmasked = np.mean((gen_np[unmasked] - gt_np[unmasked]) ** 2) if unmasked.sum() > 0 else 0.0
    
    return {
        "PSNR (masked region)": psnr_masked,
        "PSNR (full image)": psnr_full,
        "SSIM (full image)": ssim_full,
        "MSE (masked region)": mse_masked,
        "MSE (unmasked region)": mse_unmasked,
    }


# ===================== Main Test Function =====================

def test_image(image_path, output_dir=None):
    """
    Test a single face image.
    
    Args:
        image_path: Path to input face image
        output_dir: Directory to save results (optional)
    """
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"🔍 Testing Image: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"❌ Error: Could not load image {image_path}")
        return None
    
    original_size = image_bgr.shape[:2]
    print(f"Original size: {original_size[1]}x{original_size[0]}")
    
    # Resize to 256x256
    image_bgr = cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    print(f"Resized to: 256x256")
    
    # Initialize face detector
    print("\n📍 Detecting face...")
    face_detector = FaceDetector()
    masked_bgr, mask_binary, face_box = face_detector.detect_and_mask(image_bgr)
    
    mask_percentage = (mask_binary.sum() / mask_binary.size) * 100
    print(f"✓ Mask applied: {mask_percentage:.1f}% of image")
    if face_box:
        print(f"✓ Face detected at: {face_box}")
    
    # Convert to RGB
    gt_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    
    # Load generator
    print("\n🧠 Loading Generator...")
    gen = Generator(in_channels=3).to(device)
    
    checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) 
                   if f.startswith('gen_') and f.endswith('.pth.tar')]
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {config.CHECKPOINT_DIR}")
        return None
    
    latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
    cp_path = os.path.join(config.CHECKPOINT_DIR, latest_cp)
    
    checkpoint = torch.load(cp_path, map_location=device)
    gen.load_state_dict(checkpoint['state_dict'])
    gen.eval()
    print(f"✓ Loaded: {latest_cp}")
    
    # Prepare input tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    masked_tensor = transform(Image.fromarray(masked_rgb)).unsqueeze(0).to(device)
    
    # Generate
    print("\n🎨 Generating reconstructed face...")
    with torch.no_grad():
        gen_tensor = gen(masked_tensor)
    
    # Convert to numpy [0, 1]
    gen_np = gen_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    gen_np = (gen_np * 0.5) + 0.5  # [-1, 1] -> [0, 1]
    gen_np = np.clip(gen_np, 0, 1)
    
    gt_np = gt_rgb.astype(np.float32) / 255.0
    masked_np = masked_rgb.astype(np.float32) / 255.0
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    metrics = compute_metrics(gen_np, gt_np, mask_binary)
    
    # Print results
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS:")
    print("="*60)
    for metric, value in metrics.items():
        if "PSNR" in metric:
            print(f"  {metric}:\t{value:.2f} dB")
        else:
            print(f"  {metric}:\t{value:.6f}")
    print("="*60)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(gt_np)
    axes[0].set_title("Ground Truth\n(Original)", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title(f"Mask Region\n({mask_percentage:.1f}% masked)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(masked_np)
    axes[2].set_title("Masked Input\n(OpenCV DNN)", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(gen_np)
    axes[3].set_title("Generated\n(Reconstructed)", fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(f"Face Reconstruction Test - {os.path.basename(image_path)}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save comparison figure
        fig_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison saved: {fig_path}")
        
        # Save individual images
        gen_bgr = cv2.cvtColor((gen_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_generated.png"), gen_bgr)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_masked.png"), masked_bgr)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.png"), image_bgr)
        
        # Save metrics to text file
        with open(os.path.join(output_dir, f"{base_name}_metrics.txt"), 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write("="*50 + "\n")
            for metric, value in metrics.items():
                if "PSNR" in metric:
                    f.write(f"{metric}: {value:.2f} dB\n")
                else:
                    f.write(f"{metric}: {value:.6f}\n")
        
        print(f"✅ All results saved to: {output_dir}")
    
    plt.show()
    
    return {
        "generated": gen_np,
        "ground_truth": gt_np,
        "masked": masked_np,
        "mask": mask_binary,
        "metrics": metrics
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face reconstruction on a single image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input face image")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    result = test_image(args.image, args.output)
    
    if result:
        print("\n✅ Test completed successfully!")
