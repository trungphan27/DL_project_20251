
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two batches of images.
    Images should be in range [0, 1] and format (B, C, H, W).
    """
    # Convert tensor to numpy
    img1_np = img1.cpu().detach().permute(0, 2, 3, 1).numpy()
    img2_np = img2.cpu().detach().permute(0, 2, 3, 1).numpy()
    
    batch_ssim = 0
    for i in range(img1_np.shape[0]):
        # data_range=1.0 because images are normalized [0,1] or we assume standard range
        # win_size needs to be odd and smaller than image side, default is usually fine but let's be safe for 128x128
        batch_ssim += ssim(img1_np[i], img2_np[i], data_range=1.0, channel_axis=2, win_size=7)
        
    return batch_ssim / img1_np.shape[0]

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two batches of images.
    """
    img1_np = img1.cpu().detach().permute(0, 2, 3, 1).numpy()
    img2_np = img2.cpu().detach().permute(0, 2, 3, 1).numpy()
    
    batch_psnr = 0
    for i in range(img1_np.shape[0]):
        batch_psnr += psnr(img1_np[i], img2_np[i], data_range=1.0)
        
    return batch_psnr / img1_np.shape[0]

def save_sample_images(inputs, targets, outputs, epoch, batch_idx, save_dir):
    """
    Save grid of Input (Masked) | Generated (Reconstructed) | Target (Ground Truth)
    """
    with torch.no_grad():
        # Take first 4 items from batch
        N = min(inputs.size(0), 4)
        inputs = inputs[:N]
        targets = targets[:N]
        outputs = outputs[:N]
        
        # Concatenate vertically
        # (B, C, H, W) -> Grid
        
        fig, axes = plt.subplots(N, 3, figsize=(10, N * 3))
        if N == 1:
            axes = [axes] # Handle single case
            
        for i in range(N):
            # Helper to denormalize and plot
            def process(t):
                img = t[i].cpu().permute(1, 2, 0).numpy()
                img = (img * 0.5) + 0.5 # [-1, 1] -> [0, 1]
                return np.clip(img, 0, 1)

            if N > 1:
                ax_row = axes[i]
            else:
                ax_row = axes

            ax_row[0].imshow(process(inputs))
            ax_row[0].set_title("Masked Input")
            ax_row[0].axis("off")
            
            ax_row[1].imshow(process(outputs))
            ax_row[1].set_title("Generated")
            ax_row[1].axis("off")
            
            ax_row[2].imshow(process(targets))
            ax_row[2].set_title("Ground Truth")
            ax_row[2].axis("off")
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}.png"))
        plt.close()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Loading checkpoint {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we want to update weight decay or LR manually after loading
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
import config
