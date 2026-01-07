
import torch
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model import Generator
from detect import MaskDetector


def infer(image_path, gen_model=None, detector=None):
    """
    Performs inference on a single image.
    
    Args:
        image_path (str): Path to the input image (can be masked or unmasked).
        gen_model (torch.nn.Module, optional): Loaded Generator model.
        detector (MaskDetector, optional): Loaded Detector for applying mask.
        
    Returns:
        tuple: (detect_viz_img, gt_img, generated_img)
        All images are numpy arrays in BGR format (256x256).
    """
    
    # 1. Initialize models if not provided
    if detector is None:
        detector = MaskDetector()
        
    if gen_model is None:
        gen_model = Generator(in_channels=3).to(config.DEVICE)
        checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) 
                       if f.startswith('gen_') and f.endswith('.pth.tar')]
        if checkpoints:
            latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            cp_path = os.path.join(config.CHECKPOINT_DIR, latest_cp)
            checkpoint = torch.load(cp_path, map_location=config.DEVICE)
            gen_model.load_state_dict(checkpoint['state_dict'])
            gen_model.eval()
        else:
            print("Warning: No checkpoints found for Generator. Using random weights.")
            gen_model.eval()

    # 2. Process Input
    gt_img, box = detector.detect_mask(image_path)  # gt_img is BGR
    
    # Create visualization with detected region
    detect_viz_img = gt_img.copy()
    if box:
        x, y, w, h = box
        cv2.rectangle(detect_viz_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Create masked input for GAN
    masked_input_bgr = gt_img.copy()
    if box:
        masked_input_bgr = detector.apply_blackout(masked_input_bgr, box)
        
    # Resize to model input size (256x256)
    gt_small = cv2.resize(gt_img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    masked_small = cv2.resize(masked_input_bgr, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    detect_small = cv2.resize(detect_viz_img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    # 3. Run Generator
    masked_pil = Image.fromarray(cv2.cvtColor(masked_small, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(masked_pil).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        output_tensor = gen_model(input_tensor)
        
    # 4. Post-process Output
    gen_img_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    gen_img_np = (gen_img_np * 0.5) + 0.5  # [-1, 1] -> [0, 1]
    gen_img_np = np.clip(gen_img_np, 0, 1)
    gen_img_np = (gen_img_np * 255).astype(np.uint8)
    generated_img = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)
    
    return detect_small, gt_small, generated_img


if __name__ == "__main__":
    # Test on images from with_mask folder or custom test images
    test_dir = os.path.join(config.DATA_ROOT, "celeba_hq_256", "with_mask")
    output_dir = os.path.join(config.RESULT_DIR, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Looking for test images in: {test_dir}")
    print(f"Model input size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    
    if os.path.exists(test_dir):
        files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:10]  # Test first 10
        
        if not files:
            print("No images found in test directory.")
        else:
            # Initialize models once
            detector = MaskDetector()
            gen_model = Generator(in_channels=3).to(config.DEVICE)
            
            checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) 
                           if f.startswith('gen_') and f.endswith('.pth.tar')]
            if checkpoints:
                latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
                cp_path = os.path.join(config.CHECKPOINT_DIR, latest_cp)
                checkpoint = torch.load(cp_path, map_location=config.DEVICE)
                gen_model.load_state_dict(checkpoint['state_dict'])
                gen_model.eval()
                print(f"Loaded generator from {latest_cp}")
            else:
                print("Warning: No checkpoints found. Using random weights.")
                gen_model.eval()
                
            for file_name in files:
                file_path = os.path.join(test_dir, file_name)
                print(f"Processing {file_name}...")
                
                try:
                    det_img, gt, gen = infer(file_path, gen_model, detector)
                    
                    base_name = os.path.splitext(file_name)[0]
                    
                    # Save individual results
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_detect.jpg"), det_img)
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_groundtruth.jpg"), gt)
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_generated.jpg"), gen)
                    
                    # Save combined view
                    combined = np.hstack((det_img, gt, gen))
                    cv2.imwrite(os.path.join(output_dir, f"{base_name}_combined.jpg"), combined)
                    
                    print(f"  Saved results to {output_dir}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    
    else:
        print(f"Test directory not found: {test_dir}")
