import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def psnr(x, y):
    mse = F.mse_loss(x, y)
    return 10 * torch.log10(1.0 / mse)

def compute_metrics(fake, real, mask):
    hole = 1 - mask

    return {
        "psnr_masked": psnr(fake*hole, real*hole).item(),
        "psnr_full": psnr(fake, real).item(),
        "mse_masked": F.mse_loss(fake*hole, real*hole).item(),
        "mse_unmasked": F.mse_loss(fake*mask, real*mask).item(),
        "ssim": ssim(
            fake.clamp(0,1).cpu().numpy()[0].transpose(1,2,0),
            real.clamp(0,1).cpu().numpy()[0].transpose(1,2,0),
            channel_axis=2,
            data_range=1.0
        )
    }
