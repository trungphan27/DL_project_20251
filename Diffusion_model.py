import os
import io
import modal

app = modal.App("vq-ldm-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "diffusers==0.31.0",
        "accelerate==0.33.0",
        "numpy==1.26.4",
        "matplotlib",
        "Pillow",
        "einops",
        "wandb",
        "pytorch-lightning",
        "scipy==1.10.1",
        "lpips==0.1.4",
        "scikit-image==0.22.0",
    )
)

working_volume = modal.Volume.from_name("my-volume", create_if_missing=False)
mount_dir = '/mnt'

if not modal.is_local():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    import wandb
    from datetime import datetime
    from einops import rearrange
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR
    import math
    from functools import partial
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from diffusers import DDPMScheduler, UNet2DModel
    from torchvision import transforms
    from scipy import linalg
    from torchvision.models import inception_v3
    from torchvision import transforms
    import lpips
    from skimage.metrics import structural_similarity as ssim
    import pickle
    executor = ThreadPoolExecutor(max_workers=1)
else:
    import torch 
    import torch.nn as nn
    import pickle
    F = None
    np = None
    DataLoader = None
    AdamW = None
    LambdaLR = None
    wandb = None
    datetime = None
    rearrange = None
    Image = None
    from torch.utils.data import Dataset
    AdamW = None
    LambdaLR = None
    math = None
    partial = None
    plt = None
    tqdm = None
    ThreadPoolExecutor = None
    DDPMScheduler = None
    UNet2DModel = None
    transforms = None

CONFIG = {
    'gpu': 'A10G',  # or "A100", "T4", etc. A10G
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Paths (Modal-specific)
    'vqvae_ckpt': f'{mount_dir}/ldm/input/vqgan-f8/model.ckpt',
    'cache_dir': f'{mount_dir}/ldm/input/latent_cache',
    'checkpoint_dir': f'{mount_dir}/ldm/output/checkpoints',
    'sample_dir': f'{mount_dir}/ldm/output/samples',
    'val_data': f'{mount_dir}/ldm/input/test_dataset.npy',
    
    # Model architecture
    'image_size': 256,
    'latent_size': 64,
    'latent_channels': 3,
    'compression_factor': 8,
    'model_channels': 192,
    'num_res_blocks': 2,
    'channel_mult': (1, 2, 3, 4),
    'attention_resolutions': (16, 8),
    'num_heads': 8,
    'dropout': 0.1,
    
    # Training
    'batch_size': 64,
    'gradient_accumulation_steps': 1,
    'num_epochs': 50,
    'learning_rate': 14e-5,
    'weight_decay': 0.01,
    
    # Diffusion
    'timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'beta_schedule': 'linear',
    
    # Sampling
    'sample_every': 2,
    'save_every': 2,
    'num_sampling_steps': 50,
    'num_sample_images': 4,
    
    # Optimization
    'num_workers': 4,
    'use_amp': True,
    'max_grad_norm': 1.0,
    'lr_warmup_steps': 1000,
    'inpainting_weight': 7.0,
}


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        return betas.numpy()
    else:
        raise ValueError(f"Unknown schedule '{schedule}'")

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    else:
        raise NotImplementedError()

    ddim_timesteps = np.clip(ddim_timesteps, 0, num_ddpm_timesteps - 1)
    if verbose:
        print(f'DDIM timesteps: {len(ddim_timesteps)} steps')
    return ddim_timesteps

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    ddim_timesteps = np.clip(ddim_timesteps, 0, len(alphacums) - 1).astype(int)
    
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    with np.errstate(divide='ignore', invalid='ignore'):
        sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        sigmas = np.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0)
    
    return sigmas, alphas, alphas_prev

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=min(num_groups, in_channels), 
                       num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    return x * torch.sigmoid(x)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

#---------------------------------------------------------------------------
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x + h_

class Upsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, 
                 dropout=0.0, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels,
                                             kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, 
                 in_channels, resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch,
                                kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                 2*z_channels if double_z else z_channels,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], None)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Middle
        h = hs[-1]
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # End
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # Compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in,
                                kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                        out_channels=block_out,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # Prepend to get consistent order

        # End
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, None)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # End
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # Reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # Distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Compute loss
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, (torch.zeros(1), torch.zeros(1), min_encoding_indices)

class VQModel(nn.Module):
    def __init__(
        self,
        ddconfig,
        n_embed=8192,
        embed_dim=3,
    ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_last_layer(self):
        return self.decoder.conv_out.weight


def load_vqgan_from_checkpoint(ckpt_path, device='cuda'):
    print(f"Loading checkpoint from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in sd:
        sd = sd["state_dict"]
    
    # Detect configuration from checkpoint weights
    # Check encoder.down modules to determine ch_mult
    encoder_keys = [k for k in sd.keys() if k.startswith('encoder.down')]
    num_down_blocks = max([int(k.split('.')[2]) for k in encoder_keys if 'down' in k]) + 1
    
    print(f"✓ Detected {num_down_blocks} downsampling blocks")
    
    # Map downsampling blocks to ch_mult
    if num_down_blocks == 3:
        ch_mult = [1, 2, 4]  # f=8
        compression_factor = 8
    elif num_down_blocks == 4:
        ch_mult = [1, 2, 4, 4]  # f=16 or could be [1,2,4,8]
        compression_factor = 16
    elif num_down_blocks == 2:
        ch_mult = [1, 2]  # f=4
        compression_factor = 4
    else:
        # Default to f=16
        ch_mult = [1, 2, 4, 4]
        compression_factor = 16
    
    print(f"✓ Using ch_mult: {ch_mult} (compression factor f={compression_factor})")
    
    ddconfig = {
        "double_z": False,
        "z_channels": 3,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": ch_mult,
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    
    model = VQModel(
        ddconfig=ddconfig,
        n_embed=8192,
        embed_dim=3,
    )
    
    missing, unexpected = model.load_state_dict(sd, strict=False)
    
    print(f"✓ Loaded VQGAN (f={compression_factor})")
    if missing:
        print(f"  Missing keys: {len(missing)}")
        if len(missing) < 10:
            for k in missing:
                print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) < 10:
            for k in unexpected:
                print(f"    - {k}")
    
    model = model.to(device)
    model.eval()
    model.compression_factor = compression_factor
    
    return model

#---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 up=False, down=False):
        super().__init__()
        out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            Normalize(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        
        self.out_layers = nn.Sequential(
            Normalize(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )

        self.skip_connection = nn.Identity() if out_channels == channels else \
                              nn.Conv2d(channels, out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        
        h = h + emb_out
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class UNetModel(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels,
                 num_res_blocks, attention_resolutions, dropout, channel_mult,
                 **kwargs):
        super().__init__()

        self.model_channels = model_channels
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input blocks
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, 
                                  out_channels=mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttnBlock(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, down=True)
                ))
                input_block_chans.append(ch)
                ds *= 2

        # Middle
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout),
            AttnBlock(ch),
            ResBlock(ch, time_embed_dim, dropout),
        )

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout,
                                  out_channels=model_channels * mult)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttnBlock(ch))
                if level and i == num_res_blocks:
                    layers.append(ResBlock(ch, time_embed_dim, dropout, up=True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            Normalize(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        h = self.middle_block(h, emb)
        
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        
        return self.out(h)


class DiffusionWrapper(nn.Module):
    def __init__(self, unet_config):
        super().__init__()
        self.diffusion_model = UNetModel(**unet_config)

    def forward(self, x, t, c_concat):
        xc = torch.cat([x] + c_concat, dim=1)
        return self.diffusion_model(xc, t)

class LatentDiffusionModel(nn.Module):
    def __init__(self, unet_config, timesteps, 
                 beta_start, beta_end, inpainting_weight=10):
        super().__init__()
        
        self.model = DiffusionWrapper(unet_config)
        
        # VQ-VAE (frozen)
        self.first_stage_model = load_vqgan_from_checkpoint(CONFIG['vqvae_ckpt'], device='cuda')
        self.first_stage_model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        
        # Register schedule
        betas = make_beta_schedule("linear", timesteps, beta_start, beta_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.inpainting_weight = inpainting_weight

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    def apply_model(self, x_noisy, t, cond):
        return self.model(x_noisy, t, [cond])

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, cond, t, mask=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        
        loss_map = F.mse_loss(model_output, noise, reduction='none')
        
        if mask is not None:
            weight_map = torch.ones_like(mask)
            weight_map = weight_map + (self.inpainting_weight - 1.0) * mask

            weighted_loss = loss_map * weight_map
            loss = weighted_loss.mean()

            mask_loss = (loss_map * mask).sum() / (mask.sum() + 1e-8)
            background_loss = (loss_map * (1 - mask)).sum() / ((1 - mask).sum() + 1e-8)
            
            return loss, mask_loss, background_loss
        else:
            loss = loss_map.mean()
            return loss, loss, loss
        
#---------------------------------------------------------------------------
class DDIMSampler:
    def __init__(self, model):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps

    def make_schedule(self, ddim_num_steps, eta=0.):
        self.ddim_timesteps = make_ddim_timesteps("uniform", ddim_num_steps, 
                                                   self.ddpm_num_timesteps, verbose=False)
        alphacums = self.model.alphas_cumprod.cpu().numpy()
        
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums, self.ddim_timesteps, eta, verbose=False)
        
        self.ddim_sigmas = torch.tensor(ddim_sigmas).to(self.model.betas.device)
        self.ddim_alphas = torch.tensor(ddim_alphas).to(self.model.betas.device)
        self.ddim_alphas_prev = torch.tensor(ddim_alphas_prev).to(self.model.betas.device)
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas)

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning):
        self.make_schedule(ddim_num_steps=S, eta=0.)
        device = self.model.betas.device

        img = torch.randn((batch_size, *shape), device=device)
        
        timesteps = np.flip(self.ddim_timesteps).copy()
        
        for i, step in enumerate(tqdm(timesteps, desc='DDIM Sampling')):
            index = len(timesteps) - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            e_t = self.model.apply_model(img, ts, conditioning)

            a_t = self.ddim_alphas[index]
            a_prev = self.ddim_alphas_prev[index]
            sigma_t = self.ddim_sigmas[index]
            sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index]

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()

            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

            noise = sigma_t * torch.randn_like(img)
            
            img = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
        return img
    

class FastCachedDataset(Dataset):   
    def __init__(self, cache_file, split='train'): # train/val
        self.split = split
        
        print(f"Loading cached latents from {cache_file}...")
        self.latent_cache = torch.load(cache_file)
        print(f"Loaded {len(self.latent_cache)} cached latents")

        self.latent_size = list(self.latent_cache.values())[0].shape[-1]

        self.mask_cache = self._build_masks()
    
    def _build_masks(self):
        print(f"Pre-computing masks at {self.latent_size}x{self.latent_size}...")
        
        mask_cache = {}
        
        for idx in range(len(self.latent_cache)):
            mask = np.zeros((self.latent_size, self.latent_size), dtype=np.float32)
            mask_top = int(self.latent_size * 0.55)
            mask_bottom = int(self.latent_size * 0.95)
            mask_left = int(self.latent_size * 0.20)
            mask_right = int(self.latent_size * 0.80)
            mask[mask_top:mask_bottom, mask_left:mask_right] = 1.0
            
            mask = torch.from_numpy(mask[None, ...])
            mask_cache[idx] = mask
        
        print(f"✓ Masks computed!")
        return mask_cache
    
    def __len__(self):
        return len(self.latent_cache)
    
    def __getitem__(self, idx):
        z = self.latent_cache[idx]
        mask = self.mask_cache[idx]
        
        z_masked = z * (1.0 - mask)
        
        return {
            "latent": z,
            "mask": mask,
            "masked_latent": z_masked
        }


def save_checkpoint(model, optimizer, scaler, step, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'global_step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': CONFIG
    }
    path = f"{CONFIG['checkpoint_dir']}/checkpoint_epoch_{epoch+1:04d}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    all_ckpts = sorted([f for f in os.listdir(CONFIG['checkpoint_dir']) 
                        if f.startswith('checkpoint_epoch_')])
    if len(all_ckpts) > 5:
        for old_ckpt in all_ckpts[:-5]:
            os.remove(os.path.join(CONFIG['checkpoint_dir'], old_ckpt))
        print(f"Cleaned up old checkpoints")
    working_volume.commit()

@torch.no_grad()
def sample_images_cached(model, val_loader, epoch):
    model.eval()
    device = CONFIG['device']
    
    batch = next(iter(val_loader))
    z = batch['latent'][:CONFIG['num_sample_images']].to(device)
    masks = batch['mask'][:CONFIG['num_sample_images']].to(device)
    z_masked = batch['masked_latent'][:CONFIG['num_sample_images']].to(device)
    
    c = torch.cat([z_masked, masks], dim=1)

    sampler = DDIMSampler(model)
    shape = z.shape[1:]
    samples = sampler.sample(
        S=CONFIG['num_sampling_steps'],
        batch_size=CONFIG['num_sample_images'],
        shape=shape,
        conditioning=c
    )

    def decode_quantized_latent(model, z):
        """Decode already-quantized latent"""
        z = model.first_stage_model.post_quant_conv(z)
        return model.first_stage_model.decoder(z)
    
    images = decode_quantized_latent(model, z)
    masked_images = decode_quantized_latent(model, z_masked)
    x_samples = decode_quantized_latent(model, samples)

    def denorm(x):
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    fig, axes = plt.subplots(4, CONFIG['num_sample_images'], figsize=(CONFIG['num_sample_images']*3, 12))
    
    for i in range(CONFIG['num_sample_images']):
        axes[0, i].imshow(denorm(images[i]).permute(1, 2, 0).cpu())
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(denorm(masked_images[i]).permute(1, 2, 0).cpu())
        axes[1, i].set_title('Masked')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(denorm(x_samples[i]).permute(1, 2, 0).cpu())
        axes[2, i].set_title('Inpainted')
        axes[2, i].axis('off')
        
        axes[3, i].imshow(masks[i, 0].cpu(), cmap='gray')
        axes[3, i].set_title('Mask')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    save_path = f"{CONFIG['sample_dir']}/epoch_{epoch:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    wandb.log({
        "samples": wandb.Image(save_path, caption=f"Epoch {epoch}"),
        "epoch": epoch
    })
    
    plt.close()
    
    print(f"✓ Saved samples: {save_path}")
    
    model.train()


@app.function(
    image=image,
    gpu=CONFIG['gpu'],
    timeout=86400,
    volumes={mount_dir:working_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=1.0,
        initial_delay=10.0,
    ),
)
def train():
    print("Starting training on Modal...")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    wandb.login(key=os.environ["WANDB_API_KEY"])

    os.makedirs(CONFIG['sample_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    checkpoint_files = [f for f in os.listdir(CONFIG['checkpoint_dir']) 
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    start_epoch = 0
    resume_step = 0
    checkpoint = None
    
    if checkpoint_files:
        latest_ckpt = sorted(checkpoint_files)[-1]
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], latest_ckpt)
        print(f"Found checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        resume_step = checkpoint.get('global_step', 0)
        
        print(f"Will resume from epoch {start_epoch}, step {resume_step}")
    
    run_name = f"ldm-modal-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if start_epoch > 0:
        run_name += f"-resume-e{start_epoch}"
    
    wandb.init(
        project="latent-diffusion-inpainting",
        name=run_name,
        config=CONFIG,
        resume="allow",
        id=checkpoint.get('wandb_id') if checkpoint else None,
    )

    device = torch.device("cuda")
    unet_config = {
        'image_size': CONFIG['latent_size'],
        'in_channels': 7,
        'out_channels': 3,
        'model_channels': CONFIG['model_channels'],
        'attention_resolutions': CONFIG['attention_resolutions'],
        'num_res_blocks': CONFIG['num_res_blocks'],
        'channel_mult': CONFIG['channel_mult'],
        'dropout': CONFIG['dropout'],
    }
    
    model = LatentDiffusionModel(
        unet_config=unet_config,
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    ).to(device)
    
    print("Model created")
    
    # Create datasets
    train_dataset = FastCachedDataset(
        cache_file=f"{CONFIG['cache_dir']}/train.pt",
        split='train'
    )
    
    val_dataset = FastCachedDataset(
        cache_file=f"{CONFIG['cache_dir']}/val.pt",
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    optimizer = AdamW(
        model.model.parameters(),
        lr=CONFIG['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=CONFIG['weight_decay'],
    )
    
    def lr_lambda(step):
        warmup_steps = CONFIG['lr_warmup_steps']
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (len(train_loader) * CONFIG['num_epochs'] - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if CONFIG['use_amp'] else None
    
    # Training loop
    global_step = 0

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
        if scaler and checkpoint['scaler_state_dict']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        for _ in range(resume_step):
            scheduler.step()
        global_step = resume_step
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        model.train()
        model.first_stage_model.eval()
        
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            z = batch['latent'].to(device)
            mask = batch['mask'].to(device)
            z_masked = batch['masked_latent'].to(device)
            
            c = torch.cat([z_masked, mask], dim=1)
            t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=device)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    loss, mask_loss, bg_loss = model.p_losses(z, c, t, mask)
                    loss = loss / CONFIG['gradient_accumulation_steps']
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(),
                        max_norm=CONFIG['max_grad_norm']
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    
                    wandb.log({
                        "train/loss": loss.item() * CONFIG['gradient_accumulation_steps'],
                        "train/mask_loss": mask_loss.item(),
                        "train/bg_loss": bg_loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    })
                    
                    global_step += 1
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} "
                      f"Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % CONFIG['save_every'] == 0:
            executor.submit(save_checkpoint, model, optimizer, scaler, global_step, epoch, avg_loss)
        
        if (epoch + 1) % CONFIG['sample_every'] == 0:
            sample_images_cached(model, val_loader, epoch + 1)
        
    print("Training complete!")
    wandb.finish()

class RePaintInpainter:
    def __init__(self, model_id='google/ddpm-celebahq-256', device='cuda'):
        self.device = device

        print(f"Loading pretrained model: {model_id}")
        self.model = UNet2DModel.from_pretrained(model_id).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(model_id)

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        if hasattr(self.scheduler, 'betas'):
            self.scheduler.betas = self.scheduler.betas.to(device)

        self.model.eval()
        
        print(f"Model loaded. Number of timesteps: {self.scheduler.config.num_train_timesteps}")
    
    def inpaint(self, image, mask, num_inference_steps=250, jump_length=10, jump_n_sample=10, seed=None):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        batch_size = image.shape[0]

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        x_t = torch.randn_like(image).to(self.device)
        x_known = image * (1 - mask)
        x_t = x_known + x_t * mask

        t_start = timesteps[0]
        alpha_prod = self.scheduler.alphas_cumprod[t_start].to(self.device)
        noise = torch.randn_like(x_known)
        x_known_noisy = torch.sqrt(alpha_prod) * x_known + torch.sqrt(1 - alpha_prod) * noise
        x_t = x_known_noisy * (1 - mask) + x_t * mask
        
        print(f"Running RePaint inpainting...")
        print(f"Resampling: {'Enabled' if jump_length > 0 else 'Disabled'}")
        print(f"Jump length: {jump_length}, Jump n_sample: {jump_n_sample}")
        print(f"Timesteps: {timesteps[0]} -> {timesteps[-1]}")
        print(f"Mask coverage: {mask.sum().item() / mask.numel() * 100:.1f}%")

        if jump_length > 0 and jump_n_sample > 0:
            schedule = self._get_jump_schedule(
                len(timesteps), jump_length, jump_n_sample
            )
            print(f"Total steps (with resampling): {len(schedule)}")
        else:
            schedule = [(i, i+1 if i+1 < len(timesteps) else i) 
                       for i in range(len(timesteps))]
            print(f"Total steps (no resampling): {len(schedule)}")
        
        with tqdm(total=len(schedule), desc="RePaint") as pbar:
            for step_idx, (t_idx, t_next_idx) in enumerate(schedule):
                t = timesteps[t_idx] if t_idx < len(timesteps) else timesteps[-1]
                t_next = timesteps[t_next_idx] if t_next_idx < len(timesteps) else timesteps[-1]
                
                is_forward = t_next > t
                is_backward = t_next < t
                
                if is_backward:
                    x_t = self._backward_step(x_t, image, mask, t, t_next, batch_size)
                    pbar.set_postfix({'mode': 'denoise', 't': t.item()})
                    
                elif is_forward:
                    x_t = self._forward_step(x_t, image, mask, t, t_next)
                    pbar.set_postfix({'mode': 'resample', 't': t.item()})
                
                pbar.update(1)
        
        x_final = image * (1 - mask) + x_t * mask
        
        return x_final
    
    def _backward_step(self, x_t, x_known, mask, t, t_next, batch_size):
        t_tensor = t.unsqueeze(0).expand(batch_size).to(self.device)

        with torch.no_grad():
            noise_pred = self.model(x_t, t_tensor).sample
        
        x_t_next_unknown = self.scheduler.step(noise_pred, t, x_t).prev_sample

        if t_next > 0:
            alpha_prod = self.scheduler.alphas_cumprod[t_next].to(self.device)
            noise = torch.randn_like(x_known)
            x_known_noisy = (torch.sqrt(alpha_prod) * x_known * (1 - mask) + 
                            torch.sqrt(1 - alpha_prod) * noise * (1 - mask))
        else:
            x_known_noisy = x_known * (1 - mask)

        x_t_next = x_known_noisy + x_t_next_unknown * mask
        
        return x_t_next
    
    def _forward_step(self, x_t, x_known, mask, t, t_next):
        alpha_prod_t_next = self.scheduler.alphas_cumprod[t_next].to(self.device)
        alpha_prod_t = self.scheduler.alphas_cumprod[t].to(self.device)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        noise = torch.randn_like(x_t)

        x_t_next = torch.sqrt(alpha_prod_t_next / alpha_prod_t) * x_t + \
                   torch.sqrt(beta_prod_t_next - beta_prod_t * alpha_prod_t_next / alpha_prod_t) * noise
        
        noise_known = torch.randn_like(x_known)
        x_known_noisy = (torch.sqrt(alpha_prod_t_next) * x_known * (1 - mask) +
                        torch.sqrt(1 - alpha_prod_t_next) * noise_known * (1 - mask))
        
        x_t_next = x_known_noisy + x_t_next * mask
        
        return x_t_next
    
    def _get_jump_schedule(self, num_steps, jump_length, jump_n_sample):
        schedule = []
        t_idx = 0
        
        while t_idx < num_steps:
            remaining_steps = num_steps - t_idx
            
            if remaining_steps > jump_length and jump_n_sample > 0:
                for _ in range(jump_n_sample):
                    for j in range(jump_length):
                        if t_idx + j < num_steps - 1:
                            schedule.append((t_idx + j, t_idx + j + 1))

                    if t_idx + jump_length < num_steps:
                        for j in range(jump_length - 1, -1, -1):
                            if t_idx + j < num_steps - 1:
                                schedule.append((t_idx + j + 1, t_idx + j))
                
                t_idx += jump_length
            else:
                for j in range(t_idx, num_steps - 1):
                    schedule.append((j, j + 1))
                break
        
        return schedule
    
    def _denoise_step_manual(self, x_t, noise_pred, t, t_next):
        if isinstance(t, int):
            alpha_prod_t = self.scheduler.alphas_cumprod[t].to(self.device)
        else:
            alpha_prod_t = self.scheduler.alphas_cumprod[t.cpu().item()].to(self.device)
        
        if isinstance(t_next, int) and t_next >= 0:
            alpha_prod_t_next = self.scheduler.alphas_cumprod[t_next].to(self.device)
        elif hasattr(t_next, 'cpu') and t_next.cpu().item() >= 0:
            alpha_prod_t_next = self.scheduler.alphas_cumprod[t_next.cpu().item()].to(self.device)
        else:
            alpha_prod_t_next = torch.tensor(1.0).to(self.device)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next

        pred_x0 = (x_t - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        pred_x_next = (torch.sqrt(alpha_prod_t_next) * pred_x0 + 
                      torch.sqrt(beta_prod_t_next) * noise_pred)

        if (isinstance(t_next, int) and t_next > 0) or (hasattr(t_next, 'item') and t_next.item() > 0):
            variance = (beta_prod_t_next / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_next)
            variance = torch.clamp(variance, min=1e-20)
            noise = torch.randn_like(x_t)
            pred_x_next = pred_x_next + torch.sqrt(variance) * noise
        
        return pred_x_next

@app.function(
    image=image,
    gpu="T4", 
    volumes={mount_dir:working_volume},
    timeout=600,
)
def run_combined_inference(
    ldm_checkpoint_path: str,
    ddpm_model_id: str = 'google/ddpm-celebahq-256',
    num_samples: int = 8,
    ldm_sampling_steps: int = 50,
    ddpm_sampling_steps: int = 50,
    save_path: str = "/checkpoints/comparison_results.png",
    use_ddpm_resampling: bool = False,
    jump_length: int = 10,
    jump_n_sample: int = 10
):

    print("="*80)
    print("COMBINED LDM + DDPM INFERENCE COMPARISON")
    print("="*80)
    print(f"LDM Checkpoint: {ldm_checkpoint_path}")
    print(f"DDPM Model: {ddpm_model_id}")
    print(f"Samples: {num_samples}")
    print(f"LDM DDIM steps: {ldm_sampling_steps}")
    print(f"DDPM steps: {ddpm_sampling_steps}")
    print(f"DDPM Resampling: {use_ddpm_resampling}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ldm_checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], ldm_checkpoint_path.lstrip('/'))
    save_path = os.path.join(CONFIG['sample_dir'], save_path.lstrip('/'))
    
    print("Loading LDM checkpoint...")
    if not os.path.exists(ldm_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {ldm_checkpoint_path}")
    
    checkpoint = torch.load(ldm_checkpoint_path, map_location='cpu')
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    
    print("Building LDM model...")
    unet_config = {
        'image_size': CONFIG['latent_size'],
        'in_channels': 7,
        'out_channels': 3,
        'model_channels': CONFIG['model_channels'],
        'attention_resolutions': CONFIG['attention_resolutions'],
        'num_res_blocks': CONFIG['num_res_blocks'],
        'channel_mult': CONFIG['channel_mult'],
        'dropout': CONFIG['dropout'],
    }
    
    ldm_model = LatentDiffusionModel(
        unet_config=unet_config,
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    ).to(device)
    
    print("Loading LDM weights...")
    ldm_model.load_state_dict(checkpoint['model_state_dict'])
    ldm_model.eval()
    print("LDM model loaded and set to eval mode")
    
    print("Loading DDPM inpainter...")
    ddpm_inpainter = RePaintInpainter(model_id=ddpm_model_id, device=device)
    print("DDPM model loaded")

    print("Loading validation data...")
    val_dataset = FastCachedDataset(
        cache_file=f"{CONFIG['cache_dir']}/val.pt",
        split='val'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=num_samples,
        shuffle=True,
        num_workers=2
    )
    
    print(f"Sampling {num_samples} random images...")
    batch = next(iter(val_loader))
    
    z = batch['latent'][:num_samples].to(device)
    masks = batch['mask'][:num_samples].to(device)
    z_masked = batch['masked_latent'][:num_samples].to(device)
    
    print(f"  Latent shape: {z.shape}")
    print(f"  Mask shape: {masks.shape}")

    print(f"Running LDM DDIM sampling ({ldm_sampling_steps} steps)...")
    with torch.no_grad():
        c = torch.cat([z_masked, masks], dim=1)

        sampler = DDIMSampler(ldm_model)
        shape = z.shape[1:]
        
        ldm_samples = sampler.sample(
            S=ldm_sampling_steps,
            batch_size=num_samples,
            shape=shape,
            conditioning=c
        )
    
    print("LDM sampling complete!")

    print("Decoding LDM latents to images...")
    
    def decode_quantized_latent(model, z):
        z = model.first_stage_model.post_quant_conv(z)
        return model.first_stage_model.decoder(z)
    
    with torch.no_grad():
        gt_images = decode_quantized_latent(ldm_model, z)

        ldm_inpainted = decode_quantized_latent(ldm_model, ldm_samples)

        masks_upsampled = F.interpolate(masks, size=gt_images.shape[-2:], mode='nearest')
    
    print("LDM decoding complete!")
    
    print(f"Running RePaint ({ddpm_sampling_steps} steps)...")
    with torch.no_grad():
        ddpm_masked_images = gt_images * (1 - masks_upsampled) + (-1) * masks_upsampled

        ddpm_inpainted = ddpm_inpainter.inpaint(
            ddpm_masked_images,
            masks_upsampled,
            num_inference_steps=ddpm_sampling_steps,
            jump_length=jump_length if use_ddpm_resampling else 0,
            jump_n_sample=jump_n_sample if use_ddpm_resampling else 0
        )

    print("RePaint complete!")

    def denorm(x):
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        axes[i, 0].imshow(denorm(gt_images[i]).permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(denorm(ldm_inpainted[i]).permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 1].set_title('LDM Inpainting', fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(denorm(ddpm_inpainted[i]).permute(1, 2, 0).cpu().numpy())
        if i == 0:
            axes[i, 2].set_title('RePaint', fontsize=14, fontweight='bold')
        axes[i, 2].axis('off')

        mask_np = masks_upsampled[i, 0].cpu().numpy()
        axes[i, 3].imshow(mask_np, cmap='gray')
        if i == 0:
            axes[i, 3].set_title('Mask', fontsize=14, fontweight='bold')
        axes[i, 3].axis('off')

        axes[i, 0].text(-0.15, 0.5, f'#{i+1}', transform=axes[i, 0].transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    
    working_volume.commit()
    print("Committed to volume")
    
    with torch.no_grad():
        mask_binary = masks_upsampled > 0.5
        mask_expanded = mask_binary.expand(-1, 3, -1, -1)
        
        ldm_mse_masked = F.mse_loss(
            ldm_inpainted[mask_expanded],
            gt_images[mask_expanded]
        ).item()
        
        ldm_mse_unmasked = F.mse_loss(
            ldm_inpainted[~mask_expanded],
            gt_images[~mask_expanded]
        ).item()
        
        ldm_psnr = 10 * np.log10(4.0 / ldm_mse_masked)

        ddpm_mse_masked = F.mse_loss(
            ddpm_inpainted[mask_expanded],
            gt_images[mask_expanded]
        ).item()
        
        ddpm_mse_unmasked = F.mse_loss(
            ddpm_inpainted[~mask_expanded],
            gt_images[~mask_expanded]
        ).item()
        
        ddpm_psnr = 10 * np.log10(4.0 / ddpm_mse_masked)
    
    print("\nLDM RESULTS:")
    print(f"  MSE (masked region): {ldm_mse_masked:.6f}")
    print(f"  MSE (unmasked region): {ldm_mse_unmasked:.6f}")
    print(f"  PSNR (masked region): {ldm_psnr:.2f} dB")
    
    print("\nDDPM RESULTS:")
    print(f"  MSE (masked region): {ddpm_mse_masked:.6f}")
    print(f"  MSE (unmasked region): {ddpm_mse_unmasked:.6f}")
    print(f"  PSNR (masked region): {ddpm_psnr:.2f} dB")
    
    print("\nCOMPARISON:")
    mse_diff = ((ddpm_mse_masked - ldm_mse_masked) / ldm_mse_masked) * 100
    psnr_diff = ddpm_psnr - ldm_psnr
    print(f"  MSE difference: {mse_diff:+.2f}% (negative = LDM better)")
    print(f"  PSNR difference: {psnr_diff:+.2f} dB (positive = DDPM better)")
    
    print("\n" + "="*80)
    print("COMBINED INFERENCE COMPLETE!")
    print("="*80)
    
    return {
        'save_path': save_path,
        'ldm': {
            'mse_masked': ldm_mse_masked,
            'mse_unmasked': ldm_mse_unmasked,
            'psnr': ldm_psnr,
        },
        'ddpm': {
            'mse_masked': ddpm_mse_masked,
            'mse_unmasked': ddpm_mse_unmasked,
            'psnr': ddpm_psnr,
        },
        'num_samples': num_samples,
    }

class Testset(Dataset):
    def __init__(self, root):
        self.root = root
        self.images = np.load(root)
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx], 'RGB')
        image = self.transform(image)
        
        # mask
        size = CONFIG['image_size']
        mask = np.ones((size, size), dtype=np.float32)
        mask_top = int(size * 0.55)
        mask_bottom = int(size * 0.95)
        mask_left = int(size * 0.20)
        mask_right = int(size * 0.80)
        mask[mask_top:mask_bottom, mask_left:mask_right] = 0.0
        
        mask = torch.from_numpy(mask[None, ...])
        masked_image = image * mask
        mask_inverted = 1.0 - mask
        
        return {
            "image": image,
            "mask": mask_inverted,
        }

def compute_psnr(img1, img2, mask=None):
    """Compute PSNR between two images"""
    if mask is not None:
        mse = F.mse_loss(img1[mask], img2[mask])
    else:
        mse = F.mse_loss(img1, img2)
    
    if mse == 0:
        return float('inf')

    psnr = 10 * torch.log10(4.0 / mse)
    return psnr.item()

def compute_ssim(img1, img2, mask=None):
    img1_np = ((img1.cpu().numpy() + 1) / 2).clip(0, 1)
    img2_np = ((img2.cpu().numpy() + 1) / 2).clip(0, 1)
    
    ssim_values = []
    for i in range(img1_np.shape[0]):
        im1 = img1_np[i].transpose(1, 2, 0)
        im2 = img2_np[i].transpose(1, 2, 0)
        
        if mask is not None:
            mask_np = mask[i, 0].cpu().numpy()
            ssim_val = ssim(im1, im2, multichannel=True, channel_axis=2, 
                           data_range=1.0, win_size=11)
        else:
            ssim_val = ssim(im1, im2, multichannel=True, channel_axis=2, 
                           data_range=1.0, win_size=11)
        
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features.to(device).eval()

        self.slice1 = torch.nn.Sequential(*list(vgg.children())[:4])   # relu1_2
        self.slice2 = torch.nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = torch.nn.Sequential(*list(vgg.children())[9:16]) # relu3_3
        self.slice4 = torch.nn.Sequential(*list(vgg.children())[16:23])# relu4_3
        
        for param in self.parameters():
            param.requires_grad = False

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2

        x = self.normalize(x)
        y = self.normalize(y)

        x_relu1 = self.slice1(x)
        y_relu1 = self.slice1(y)
        
        x_relu2 = self.slice2(x_relu1)
        y_relu2 = self.slice2(y_relu1)
        
        x_relu3 = self.slice3(x_relu2)
        y_relu3 = self.slice3(y_relu2)
        
        x_relu4 = self.slice4(x_relu3)
        y_relu4 = self.slice4(y_relu3)

        loss1 = F.mse_loss(x_relu1, y_relu1)
        loss2 = F.mse_loss(x_relu2, y_relu2)
        loss3 = F.mse_loss(x_relu3, y_relu3)
        loss4 = F.mse_loss(x_relu4, y_relu4)
        
        return loss1 + loss2 + loss3 + loss4

class InceptionFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(device).eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def extract_features(self, images):
        images = (images + 1) / 2

        if images.shape[-1] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        images = self.normalize(images)
        
        with torch.no_grad():
            features = self.model(images)
        
        return features.cpu().numpy()

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def save_val_checkpoint(checkpoint_path, state):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"  💾 Checkpoint saved: {checkpoint_path}")

def load_val_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        print(f"  ✅ Checkpoint loaded: {checkpoint_path}")
        return state
    return None


@app.function(
    image=image,
    gpu="T4", 
    volumes={mount_dir:working_volume},
    timeout=18000,
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=1.0,
        initial_delay=10.0,
    ),
)
def run_ldm_validation(
    ldm_checkpoint_path: str,
    num_samples: int = 100,
    encoding_batch_size: int = 32,
    inference_batch_size: int = 4,
    ldm_sampling_steps: int = 50,
    save_path: str = "validation_images.png",
    compute_fid: bool = True,
    num_vis_samples: int = 8,
    resume_from_checkpoint: bool = True,
    checkpoint_interval: int = 10
):
    print(f"LDM Checkpoint: {ldm_checkpoint_path}")
    print(f"Total samples: {num_samples}")
    print(f"Encoding batch size: {encoding_batch_size}")
    print(f"Inference batch size: {inference_batch_size}")
    print(f"DDIM steps: {ldm_sampling_steps}")
    print(f"Compute FID: {compute_fid}")
    print(f"Resume from checkpoint: {resume_from_checkpoint}")
    print()
    
    device = CONFIG['device']
    
    ldm_checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], ldm_checkpoint_path.lstrip('/'))
    save_path = os.path.join(CONFIG['sample_dir'], save_path.lstrip('/'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint_dir = os.path.join(CONFIG['checkpoint_dir'], 'validation_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    encoded_data_path = os.path.join(checkpoint_dir, 'encoded_data.pt')
    progress_checkpoint_path = os.path.join(checkpoint_dir, 'validation_progress.pkl')

    print("Loading LDM checkpoint...")
    if not os.path.exists(ldm_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {ldm_checkpoint_path}")
    
    checkpoint = torch.load(ldm_checkpoint_path, map_location='cpu')
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    
    print("\nBuilding LDM model...")
    unet_config = {
        'image_size': CONFIG['latent_size'],
        'in_channels': 7,
        'out_channels': 3,
        'model_channels': CONFIG['model_channels'],
        'attention_resolutions': CONFIG['attention_resolutions'],
        'num_res_blocks': CONFIG['num_res_blocks'],
        'channel_mult': CONFIG['channel_mult'],
        'dropout': CONFIG['dropout'],
    }
    
    ldm_model = LatentDiffusionModel(
        unet_config=unet_config,
        timesteps=CONFIG['timesteps'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end']
    ).to(device)
    
    print("Loading LDM weights...")
    ldm_model.load_state_dict(checkpoint['model_state_dict'])
    ldm_model.eval()
    print("LDM model loaded and set to eval mode")
    
    print("Loading validation dataset...")
    
    val_dataset = Testset(CONFIG['val_data'])
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=encoding_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Loaded {len(val_dataset)} validation samples")
    

    if resume_from_checkpoint and os.path.exists(encoded_data_path):
        print("Loading pre-encoded data from checkpoint...")
        try:
            encoded_data = torch.load(encoded_data_path, map_location='cpu')
            all_latents = encoded_data['latents']
            all_masks = encoded_data['masks']
            all_images = encoded_data['images']
            print(f"Loaded {all_latents.shape[0]} pre-encoded samples")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Re-encoding from scratch...")
            encoded_data = None
    else:
        encoded_data = None
    
    if encoded_data is None:
        print("🔄 Encoding all validation images to latent space...")
        
        all_latents = []
        all_masks = []
        all_images = []
        
        num_batches_to_encode = min(
            num_samples // encoding_batch_size + (1 if num_samples % encoding_batch_size else 0), 
            len(val_loader)
        )
        total_encoded = 0
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, total=num_batches_to_encode, desc="Encoding")):
                    if total_encoded >= num_samples:
                        break
                    
                    real_images = batch['image'].to(device)
                    masks = batch['mask'].to(device)

                    z_gt, _, _ = ldm_model.first_stage_model.encode(real_images)

                    masks_latent = F.interpolate(
                        masks, 
                        size=(CONFIG['latent_size'], CONFIG['latent_size']), 
                        mode='nearest'
                    )

                    all_latents.append(z_gt.cpu())
                    all_masks.append(masks_latent.cpu())
                    all_images.append(real_images.cpu())
                    
                    total_encoded += real_images.shape[0]
            
            all_latents = torch.cat(all_latents, dim=0)[:num_samples]
            all_masks = torch.cat(all_masks, dim=0)[:num_samples]
            all_images = torch.cat(all_images, dim=0)[:num_samples]

            print("\nSaving encoded data checkpoint...")
            torch.save({
                'latents': all_latents,
                'masks': all_masks,
                'images': all_images,
            }, encoded_data_path)
            print(f"Saved to: {encoded_data_path}")
            
        except Exception as e:
            print(f"\nERROR during encoding: {e}")
            raise
    
    print(f"Encoded {all_latents.shape[0]} images")
    print(f"Latent shape: {all_latents.shape}")
    print(f"Mask shape: {all_masks.shape}")

    all_masked_latents = all_latents * (1 - all_masks)
    
    print("Setting up metric calculators...")
    vgg_loss_fn = VGGPerceptualLoss(device=device)
    print("  ✓ VGG Perceptual Loss initialized")
    
    if compute_fid:
        inception_extractor = InceptionFeatureExtractor(device=device)
        print("Inception V3 for FID initialized")
    
    progress_state = None
    if resume_from_checkpoint:
        progress_state = load_val_checkpoint(progress_checkpoint_path)
    
    if progress_state is not None:
        print(f"Resuming from batch {progress_state['last_batch_idx'] + 1}")
        start_batch_idx = progress_state['last_batch_idx'] + 1
        psnr_masked_list = progress_state['psnr_masked_list']
        psnr_full_list = progress_state['psnr_full_list']
        ssim_full_list = progress_state['ssim_full_list']
        vgg_loss_list = progress_state['vgg_loss_list']
        mse_masked_list = progress_state['mse_masked_list']
        mse_unmasked_list = progress_state['mse_unmasked_list']
        real_features_list = progress_state.get('real_features_list', [])
        fake_features_list = progress_state.get('fake_features_list', [])
        vis_data = progress_state['vis_data']
    else:
        print("Starting fresh validation")
        start_batch_idx = 0
        psnr_masked_list = []
        psnr_full_list = []
        ssim_full_list = []
        vgg_loss_list = []
        mse_masked_list = []
        mse_unmasked_list = []
        real_features_list = []
        fake_features_list = []
        vis_data = {
            'gt_images': [],
            'masked_images': [],
            'inpainted_images': [],
            'masks': []
        }
    
    sampler = DDIMSampler(ldm_model)
    num_batches = (num_samples + inference_batch_size - 1) // inference_batch_size
    
    print(f"Processing batches {start_batch_idx + 1} to {num_batches}...")
    
    try:
        for batch_idx in tqdm(range(start_batch_idx, num_batches), desc="Inference", initial=start_batch_idx, total=num_batches):
            start_idx = batch_idx * inference_batch_size
            end_idx = min(start_idx + inference_batch_size, num_samples)
            
            z_gt = all_latents[start_idx:end_idx].to(device)
            masks_latent = all_masks[start_idx:end_idx].to(device)
            z_masked = all_masked_latents[start_idx:end_idx].to(device)
            real_images = all_images[start_idx:end_idx].to(device)
            
            batch_size_actual = z_gt.shape[0]
            
            with torch.no_grad():
                c = torch.cat([z_masked, masks_latent], dim=1)
                
                shape = z_gt.shape[1:]
                z_inpainted = sampler.sample(
                    S=ldm_sampling_steps,
                    batch_size=batch_size_actual,
                    shape=shape,
                    conditioning=c
                )
                
                inpainted_images = ldm_model.first_stage_model.decode(z_inpainted)
                
                masks_upsampled = F.interpolate(
                    masks_latent, 
                    size=real_images.shape[-2:], 
                    mode='nearest'
                )
                
                masked_images = real_images * (1 - masks_upsampled) + (-1) * masks_upsampled
                
                mask_binary = masks_upsampled > 0.5
                mask_expanded = mask_binary.expand(-1, 3, -1, -1)

                # PSNR - Masked region
                psnr_masked = compute_psnr(inpainted_images, real_images, mask_expanded)
                psnr_masked_list.append(psnr_masked)
                
                # PSNR - Full image
                psnr_full = compute_psnr(inpainted_images, real_images, None)
                psnr_full_list.append(psnr_full)
                
                # SSIM
                ssim_val = compute_ssim(inpainted_images, real_images)
                ssim_full_list.append(ssim_val)
                
                # VGG Perceptual Loss
                vgg_loss = vgg_loss_fn(inpainted_images, real_images)
                vgg_loss_list.append(vgg_loss.item())
                
                # MSE
                mse_masked = F.mse_loss(
                    inpainted_images[mask_expanded],
                    real_images[mask_expanded]
                ).item()
                mse_masked_list.append(mse_masked)
                
                mse_unmasked = F.mse_loss(
                    inpainted_images[~mask_expanded],
                    real_images[~mask_expanded]
                ).item()
                mse_unmasked_list.append(mse_unmasked)
                
                # FID features
                if compute_fid:
                    real_feats = inception_extractor.extract_features(real_images)
                    fake_feats = inception_extractor.extract_features(inpainted_images)
                    real_features_list.append(real_feats)
                    fake_features_list.append(fake_feats)

                if len(vis_data['gt_images']) < num_vis_samples:
                    n_to_save = min(num_vis_samples - len(vis_data['gt_images']), batch_size_actual)
                    vis_data['gt_images'].append(real_images[:n_to_save].cpu())
                    vis_data['masked_images'].append(masked_images[:n_to_save].cpu())
                    vis_data['inpainted_images'].append(inpainted_images[:n_to_save].cpu())
                    vis_data['masks'].append(masks_upsampled[:n_to_save].cpu())

            if (batch_idx + 1) % checkpoint_interval == 0 or batch_idx == num_batches - 1:
                checkpoint_state = {
                    'last_batch_idx': batch_idx,
                    'psnr_masked_list': psnr_masked_list,
                    'psnr_full_list': psnr_full_list,
                    'ssim_full_list': ssim_full_list,
                    'vgg_loss_list': vgg_loss_list,
                    'mse_masked_list': mse_masked_list,
                    'mse_unmasked_list': mse_unmasked_list,
                    'real_features_list': real_features_list if compute_fid else [],
                    'fake_features_list': fake_features_list if compute_fid else [],
                    'vis_data': vis_data,
                }
                save_val_checkpoint(progress_checkpoint_path, checkpoint_state)
    
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        print("Progress has been saved. You can resume by running the function again.")
        raise

    avg_psnr_masked = np.mean(psnr_masked_list)
    avg_psnr_full = np.mean(psnr_full_list)
    avg_ssim_full = np.mean(ssim_full_list)
    avg_vgg_loss = np.mean(vgg_loss_list)
    avg_mse_masked = np.mean(mse_masked_list)
    avg_mse_unmasked = np.mean(mse_unmasked_list)
    
    print("\nVALIDATION RESULTS:")
    print(f"  PSNR (masked region):   {avg_psnr_masked:.2f} dB")
    print(f"  PSNR (full image):      {avg_psnr_full:.2f} dB")
    print(f"  SSIM (full image):      {avg_ssim_full:.4f}")
    print(f"  VGG Perceptual Loss:    {avg_vgg_loss:.6f}")
    print(f"  MSE (masked region):    {avg_mse_masked:.6f}")
    print(f"  MSE (unmasked region):  {avg_mse_unmasked:.6f}")
    
    # Compute FID
    if compute_fid:
        print("\nComputing FID...")
        real_features = np.concatenate(real_features_list, axis=0)
        fake_features = np.concatenate(fake_features_list, axis=0)
        fid_score = calculate_fid(real_features, fake_features)
        print(f"  FID Score:              {fid_score:.2f}")
    
    def denorm(x):
        """Denormalize from [-1, 1] to [0, 1]"""
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    gt_images = torch.cat(vis_data['gt_images'], dim=0)[:num_vis_samples]
    masked_images = torch.cat(vis_data['masked_images'], dim=0)[:num_vis_samples]
    inpainted_images = torch.cat(vis_data['inpainted_images'], dim=0)[:num_vis_samples]
    masks_vis = torch.cat(vis_data['masks'], dim=0)[:num_vis_samples]

    fig, axes = plt.subplots(num_vis_samples, 4, figsize=(16, num_vis_samples*4))
    
    if num_vis_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_vis_samples):
        axes[i, 0].imshow(denorm(gt_images[i]).permute(1, 2, 0).numpy())
        if i == 0:
            axes[i, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(denorm(masked_images[i]).permute(1, 2, 0).numpy())
        if i == 0:
            axes[i, 1].set_title('Masked Input', fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(denorm(inpainted_images[i]).permute(1, 2, 0).numpy())
        if i == 0:
            axes[i, 2].set_title('LDM Inpainting', fontsize=14, fontweight='bold')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(masks_vis[i, 0].numpy(), cmap='gray')
        if i == 0:
            axes[i, 3].set_title('Mask', fontsize=14, fontweight='bold')
        axes[i, 3].axis('off')

        axes[i, 0].text(-0.15, 0.5, f'#{i+1}', transform=axes[i, 0].transAxes,
                       fontsize=12, fontweight='bold', va='center', ha='right')

    title_text = f"LDM Validation Results\n"
    title_text += f"PSNR: {avg_psnr_masked:.2f} dB | SSIM: {avg_ssim_full:.4f} | VGG Loss: {avg_vgg_loss:.6f}"
    if compute_fid:
        title_text += f" | FID: {fid_score:.2f}"
    
    plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {save_path}")

    working_volume.commit()
    print("  ✓ Committed to volume")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)

    results = {
        'save_path': save_path,
        'num_samples': num_samples,
        'psnr_masked': avg_psnr_masked,
        'psnr_full': avg_psnr_full,
        'ssim': avg_ssim_full,
        'vgg_loss': avg_vgg_loss,
        'mse_masked': avg_mse_masked,
        'mse_unmasked': avg_mse_unmasked,
    }
    
    if compute_fid:
        results['fid'] = fid_score
    
    return results

@app.local_entrypoint()
def main():
    #train.remote()
    '''
    results = run_combined_inference.remote(
        ldm_checkpoint_path="checkpoint_epoch_0050.pt",
        ddpm_model_id='google/ddpm-celebahq-256',
        num_samples=4,
        ldm_sampling_steps=50,
        ddpm_sampling_steps=50,
        save_path="ldm_vs_ddpm_comparison.png",
        use_ddpm_resampling=True,
        jump_length=10,
        jump_n_sample=10
    )
    print(results)
    '''
    results = run_ldm_validation.remote(
        ldm_checkpoint_path="checkpoint_epoch_0050.pt",
        num_samples=6000,
        encoding_batch_size=16,
        inference_batch_size=32,
        ldm_sampling_steps=50,
        compute_fid=True, 
        num_vis_samples=8
    )
    print(results)