import torch
import numpy as np
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg


class InceptionFID(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(
            weights=weights,
            aux_logits=True    # ⚠️ BẮT BUỘC với torchvision hiện tại
        )

        # ❗ BỎ HEAD
        model.fc = torch.nn.Identity()
        model.AuxLogits = None   # ❗ TẮT AUX SAU KHI LOAD

        self.model = model.to(device).eval()

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        )

    @torch.no_grad()
    def get_acts(self, x):
        """
        x: Tensor [B,3,256,256], range [-1,1]
        """
        x = x.to(self.device)

        # [-1,1] → [0,1]
        x = (x + 1) / 2

        # resize
        x = F.interpolate(
            x, size=(299, 299),
            mode="bilinear",
            align_corners=False
        )

        # normalize ImageNet
        x = (x - self.mean) / self.std

        feats = self.model(x)

        # đảm bảo [B,2048]
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, 1)
            feats = feats.squeeze(-1).squeeze(-1)

        return feats.cpu().numpy()


def calculate_fid(real_acts, fake_acts):
    mu1 = real_acts.mean(axis=0)
    mu2 = fake_acts.mean(axis=0)

    sigma1 = np.cov(real_acts, rowvar=False)
    sigma2 = np.cov(fake_acts, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)
