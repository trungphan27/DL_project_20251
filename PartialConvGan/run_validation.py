import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from dataset.testset import Testset
from models.pconv_unet import PConvUNet
from validation.fid_utils import InceptionFID, calculate_fid


NPY_PATH = "test_dataset.npy"
CKPT = "checkpoints/G_only.pth"
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = PConvUNet()
G.load_state_dict(torch.load(CKPT, map_location=device))
G = G.to(device)
G.eval()

dataset = Testset(NPY_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

psnr_masked_list = []
psnr_full_list = []
ssim_full_list = []
mse_masked_list = []
mse_unmasked_list = []

fid_model = InceptionFID(device)
real_acts, fake_acts = [], []

with torch.no_grad():
    for batch in tqdm(loader):

        masked = batch[0].to(device)   
        mask   = batch[1].to(device)   
        real   = batch[2].to(device)   

        fake = G(masked, mask)

        composite = fake * mask + real * (1 - mask)

        fake_np = fake.cpu().numpy()
        real_np = real.cpu().numpy()
        comp_np = composite.cpu().numpy()
        mask_np = mask.cpu().numpy()
        inv_mask_np = 1.0 - mask_np

        for i in range(fake_np.shape[0]):

            psnr_masked_list.append(
                peak_signal_noise_ratio(
                    real_np[i] * mask_np[i],
                    fake_np[i] * mask_np[i],
                    data_range=2
                )
            )

            psnr_full_list.append(
                peak_signal_noise_ratio(
                    real_np[i],
                    comp_np[i],
                    data_range=2
                )
            )

            ssim_full_list.append(
                structural_similarity(
                    real_np[i].transpose(1, 2, 0),
                    comp_np[i].transpose(1, 2, 0),
                    channel_axis=2,
                    data_range=2
                )
            )

            mse_masked_list.append(
                ((real_np[i] - fake_np[i]) ** 2 * mask_np[i]).sum()
                / (mask_np[i].sum() + 1e-8)
            )

            mse_unmasked_list.append(
                ((real_np[i] - fake_np[i]) ** 2 * inv_mask_np[i]).sum()
                / (inv_mask_np[i].sum() + 1e-8)
            )

        real_fid = real * mask
        fake_fid = composite * mask

        real_acts.append(fid_model.get_acts(real_fid))
        fake_acts.append(fid_model.get_acts(fake_fid))



real_acts = np.concatenate(real_acts, axis=0)
fake_acts = np.concatenate(fake_acts, axis=0)
fid = calculate_fid(real_acts, fake_acts)

print("\n===== VALIDATION RESULT =====")
print(f"PSNR (masked region) : {np.mean(psnr_masked_list):.2f} dB")
print(f"PSNR (full image)   : {np.mean(psnr_full_list):.2f} dB")
print(f"SSIM (full image)   : {np.mean(ssim_full_list):.4f}")
print(f"MSE  (masked)       : {np.mean(mse_masked_list):.6f}")
print(f"MSE  (unmasked)     : {np.mean(mse_unmasked_list):.6f}")
print(f"FID                 : {fid:.2f}")
