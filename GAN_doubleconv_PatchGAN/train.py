import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from model import Generator, Discriminator, initialize_weights
from dataset import CelebAMaskDataset, get_transforms
from utils import save_sample_images, load_checkpoint, calculate_ssim, calculate_psnr
from loss import VGGLoss
import os
import csv
import random


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, vgg_loss, epoch, device):
    loop = tqdm(loader, leave=True)
    
    d_losses = []
    g_losses = []
    l1_losses = []
    vgg_losses = []
    ssim_scores = []
    psnr_scores = []

    for idx, (x, y) in enumerate(loop):
        x = x.to(device)  # Masked Image (input)
        y = y.to(device)  # Ground Truth (target)

        # --- Train Discriminator ---
        y_fake = gen(x)
        d_real = disc(y)
        d_fake = disc(y_fake.detach())

        d_real_loss = bce(d_real, torch.ones_like(d_real))
        d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_real_loss + d_fake_loss) / 2

        opt_disc.zero_grad()
        d_loss.backward()
        opt_disc.step()

        # --- Train Generator ---
        d_fake = disc(y_fake)
        g_fake_loss = bce(d_fake, torch.ones_like(d_fake)) * config.LAMBDA_ADV
        g_l1_loss = l1_loss(y_fake, y) * config.LAMBDA_L1
        g_vgg_loss = vgg_loss(y_fake, y) * config.LAMBDA_VGG

        g_loss = g_fake_loss + g_l1_loss + g_vgg_loss

        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()
        
        # Calculate Metrics
        with torch.no_grad():
            s = calculate_ssim(y_fake, y)
            p = calculate_psnr(y_fake, y)
            
            ssim_scores.append(s)
            psnr_scores.append(p)
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            l1_losses.append(g_l1_loss.item())
            vgg_losses.append(g_vgg_loss.item())

        loop.set_postfix(
            D=d_loss.item(), 
            G=g_loss.item(),
            L1=g_l1_loss.item(),
            VGG=g_vgg_loss.item(),
            SSIM=s,
            PSNR=p,
        )
        
        if idx % 500 == 0:
            save_sample_images(x, y, y_fake, epoch, idx, config.RESULT_DIR)
            
    return (
        sum(d_losses)/len(d_losses),
        sum(g_losses)/len(g_losses),
        sum(l1_losses)/len(l1_losses),
        sum(vgg_losses)/len(vgg_losses),
        sum(ssim_scores)/len(ssim_scores),
        sum(psnr_scores)/len(psnr_scores)
    )


def main():
    device = config.DEVICE
    
    disc = Discriminator(in_channels=3).to(device)
    gen = Generator(in_channels=3, features=[64, 128, 256, 512]).to(device)
    
    initialize_weights(disc)
    initialize_weights(gen)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE_D, betas=config.BETAS)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_G, betas=config.BETAS)
    
    # Schedulers (UNCHANGED)
    scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, mode='min', factor=0.5, patience=5, verbose=True)

    bce = nn.BCELoss()
    l1_loss = nn.L1Loss()
    vgg_loss = VGGLoss().to(device)

    # Dataset Setup - Use paired images from CelebA-HQ
    print(f"Loading dataset from:")
    print(f"  Ground Truth: {config.DATASET_DIR}")
    print(f"  Masked: {config.MASKED_DIR}")
    
    # Get file list from ground truth directory
    full_file_list = [f for f in os.listdir(config.DATASET_DIR) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(full_file_list)
    
    train_size = int(len(full_file_list) * config.TRAIN_RATIO)
    train_files = full_file_list[:train_size]
    
    train_dataset = CelebAMaskDataset(
        file_list=train_files,
        gt_dir=config.DATASET_DIR,
        masked_dir=config.MASKED_DIR,
        transforms=get_transforms()
    )
    
    print(f"Training on {len(train_dataset)} images (IMG_SIZE={config.IMG_SIZE})")

    loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Resumable Logic
    start_epoch = 0
    if config.LOAD_MODEL:
        checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.endswith('.pth.tar') and 'gen' in f]
        if checkpoints:
            latest_cp = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_epoch = int(latest_cp.split('_')[1].split('.')[0])
            
            gen_path = os.path.join(config.CHECKPOINT_DIR, f"gen_{latest_epoch}.pth.tar")
            disc_path = os.path.join(config.CHECKPOINT_DIR, f"disc_{latest_epoch}.pth.tar")
            
            if os.path.exists(gen_path) and os.path.exists(disc_path):
                load_checkpoint(gen_path, gen, opt_gen, config.LEARNING_RATE_G)
                load_checkpoint(disc_path, disc, opt_disc, config.LEARNING_RATE_D)
                start_epoch = latest_epoch + 1
                print(f"Resumed from epoch {start_epoch}")
    
    # Log File
    log_file = os.path.join(config.RESULT_DIR, "train_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "D_Loss", "G_Loss", "L1_Loss", "VGG_Loss", "SSIM", "PSNR"])

    # Training Loop
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS - 1}]")
        
        d_loss, g_loss, l1, vgg, ssim, psnr = train_fn(
            disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, vgg_loss, epoch, device
        )
        
        # Step Scheduler
        scheduler_disc.step(d_loss)
        scheduler_gen.step(g_loss)
        
        # Log
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, d_loss, g_loss, l1, vgg, ssim, psnr])
            
        print(f"Stats - G_Loss: {g_loss:.4f}, SSIM: {ssim:.4f}, PSNR: {psnr:.4f}")

        # Save Checkpoint
        if (epoch + 1) % config.SAVE_EVERY == 0:
            cp_gen = {
                "state_dict": gen.state_dict(),
                "optimizer": opt_gen.state_dict(),
            }
            torch.save(cp_gen, os.path.join(config.CHECKPOINT_DIR, f"gen_{epoch}.pth.tar"))
            
            cp_disc = {
                "state_dict": disc.state_dict(),
                "optimizer": opt_disc.state_dict(),
            }
            torch.save(cp_disc, os.path.join(config.CHECKPOINT_DIR, f"disc_{epoch}.pth.tar"))


if __name__ == "__main__":
    main()
