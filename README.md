# 🎭 Face Reconstruction Under Mask - Deep Learning Project

> **Deep Learning Course Project - Semester 20251**

**Objective**: Reconstruct masked facial regions using deep generative models.

---

## I. 👥 Team Members

| No. | Full Name        | Student ID |
| :-: | ---------------- | :--------: |
|  1  | Phan Đình Trung  |  20230093  |
|  2  | Phạm Văn Vũ Hoàn |  20235497  |
|  3  | Phan Hải Nguyên  |  20235540  |
|  4  | Lê Hoàng Nam     |  20235536  |
|  5  | Nguyễn Trung Hải |  20235495  |

---

## II. 📊 Datasets

### 2.1. Training Dataset

- **CelebA-HQ Resized 256×256**: [Kaggle Link](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)
- Total images: 30,000 high-quality face images
- Resolution: 256 × 256 pixels

### 2.2. Data Preprocessing

- Applied **OpenCV DNN Face Detector** for face detection
- Generated synthetic black rectangular masks covering the lower face region (nose, mouth, chin)
- Mask region: Lower 45% of the face bounding box (from 55% to 100% of face height)
- Dataset structure:
  - `without_mask/`: Original images (ground truth)
  - `with_mask/`: Masked images (input)

### 2.3. Cross-Dataset Evaluation & Validation

- **Flickr-Faces-HQ (FFHQ)**: [Kaggle Link](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)
- Used to evaluate model generalization on a different dataset

---

## III. 🔬 Methods

### 1. PremiumGAN (GAN with PatchGAN Discriminator)

#### 1.1. Model Architecture

**Generator (U-Net Based)**

- Encoder-decoder architecture with skip connections
- **Encoder**: 4 DoubleConv blocks with features [64, 128, 256, 512], each block contains:
  - `Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(3×3) → BatchNorm2d → ReLU`
  - MaxPool2d(2×2) for downsampling
- **Bottleneck**: DoubleConv with 1024 channels
- **Decoder**: 4 ConvTranspose2d + DoubleConv blocks with skip connections
- **Output**: `Conv2d(1×1) → Tanh()` for output range [-1, 1]

**Discriminator (PatchGAN)**

- Outputs an NxN probability grid instead of a single scalar
- **Initial**: `Conv2d(4×4, stride=2) → LeakyReLU(0.2)`
- **CNNBlocks**: 3 blocks with features [128, 256, 512]
  - `Conv2d(4×4, stride=2) → BatchNorm2d → LeakyReLU(0.2)`
- **Output**: `Conv2d(4×4, stride=1) → Sigmoid()`

**Loss Functions**

- **Adversarial Loss**: BCELoss with λ_adv = 1
- **L1 Reconstruction Loss**: λ_L1 = 100
- **VGG Perceptual Loss**: Using pretrained VGG19, extracting features from 5 layers (relu1_1 to relu5_1), λ_VGG = 10

#### 1.2. Usage Guide

##### Step 1: Install Dependencies

```bash
# Clone repository
git clone <repository_url>
cd DL_project_20251

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

##### Step 2: Download Pretrained Models

Access Google Drive and download the checkpoint files:

🔗 **Download Link**: [Google Drive - Checkpoints](https://drive.google.com/drive/folders/1Dgu5bhrxTH0oIfWEO807QzUqTYGRq0MH)

Download and place the following files into `ImprovedPremiumGAN_celeba/checkpoints/`:

- `gen_49.pth.tar` (Generator checkpoint)
- `disc_49.pth.tar` (Discriminator checkpoint)

```
ImprovedPremiumGAN_celeba/
├── checkpoints/
│   ├── gen_49.pth.tar      ← Download and place here
│   └── disc_49.pth.tar     ← Download and place here
```

##### Step 3: Prepare Dataset (if retraining)

```bash
cd ImprovedPremiumGAN_celeba

# Place CelebA-HQ dataset in celeba_hq_256/ folder
# Run script to create masked dataset
python prepare_masked_dataset.py
```

The script will automatically:

1. Copy original images to `celeba_hq_256/without_mask/`
2. Use OpenCV DNN Face Detector to detect faces
3. Create black masks covering the lower face region
4. Save masked images to `celeba_hq_256/with_mask/`

##### Step 4: Training (Optional)

```bash
cd ImprovedPremiumGAN_celeba

# Train model (supports automatic resume training)
python train.py
```

**Default Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Image Size | 256×256 |
| Batch Size | 8 |
| Epochs | 100 |
| Learning Rate (G & D) | 0.0002 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Train/Test Split | 90% / 10% |

##### Step 5: Test on Single Image

```bash
# Test with any face image
python test.py --image path/to/your/face_image.jpg

# Test and save results
python test.py --image path/to/your/face_image.jpg --output results/
```

The script will:

1. Resize image to 256×256
2. Automatically detect face and create mask
3. Run Generator to reconstruct the masked region
4. Display and save comparison results

##### Step 6: Validate on Large Dataset

```bash
# Validate on CelebA-HQ test set
python validate.py --num_images 3000 --data_dir "path/to/without_mask" --compute_fid

# Validate on FFHQ dataset (cross-dataset evaluation)
python validate.py --num_images 3000 --data_dir "path/to/ffhq" --compute_fid
```

#### 1.3. Experimental Results

##### 📈 Training Results

**Best Checkpoint: Epoch 49**

| Metric   | Value        |
| -------- | ------------ |
| D_Loss   | 0.2149       |
| G_Loss   | 17.1841      |
| L1_Loss  | 3.0853       |
| VGG_Loss | 10.1185      |
| **SSIM** | **0.8834**   |
| **PSNR** | **23.78 dB** |

**Training Statistics (50 Epochs)**

| Metric | Min      | Max      | Mean     | Std    |
| ------ | -------- | -------- | -------- | ------ |
| D_Loss | 0.2063   | 0.5850   | 0.3132   | 0.1053 |
| G_Loss | 16.91    | 33.22    | 17.81    | 2.29   |
| SSIM   | 0.7163   | 0.8834   | 0.8704   | 0.0244 |
| PSNR   | 17.10 dB | 23.78 dB | 22.90 dB | 1.08   |

**Improvement (First → Best Epoch)**

- SSIM: **+23.32%**
- PSNR: **+39.06%**
- G_Loss: **-48.27%** (reduced)

##### 📊 Validation Results (CelebA-HQ Test Set - 3000 images)

| Metric                | Value      |
| --------------------- | ---------- |
| PSNR (masked region)  | 22.01 dB   |
| PSNR (full image)     | 29.02 dB   |
| **SSIM (full image)** | **0.9299** |
| VGG Perceptual Loss   | 1.4467     |
| MSE (masked region)   | 0.006728   |
| MSE (unmasked region) | 0.000073   |
| **FID Score**         | **5.56**   |

##### 🔄 Cross-Dataset Evaluation Results (FFHQ Dataset)

| Metric                | Value      |
| --------------------- | ---------- |
| PSNR (masked region)  | 19.38 dB   |
| PSNR (full image)     | 26.37 dB   |
| **SSIM (full image)** | **0.8965** |
| VGG Perceptual Loss   | 2.2809     |
| MSE (masked region)   | 0.013072   |
| MSE (unmasked region) | 0.000210   |
| **FID Score**         | **7.05**   |

---

### 2. GANsFormer

_(To be implemented...)_

---

### 3. Pretrained Diffusion Model

_(To be implemented...)_

---

## 📁 Project Structure

```
DL_project_20251/
├── ImprovedPremiumGAN_celeba/
│   ├── checkpoints/           # Pretrained models
│   ├── models/                # OpenCV DNN Face Detector
│   ├── results/               # Training logs & sample images
│   │   ├── plots/             # Training statistics & plots
│   │   ├── train_log.csv      # Training metrics per epoch
│   │   └── validation_results.txt
│   ├── config.py              # Hyperparameters
│   ├── model.py               # Generator & Discriminator
│   ├── loss.py                # VGG Perceptual Loss
│   ├── dataset.py             # CelebA Dataset loader
│   ├── train.py               # Training script
│   ├── test.py                # Single image testing
│   ├── validate.py            # Full dataset validation
│   └── prepare_masked_dataset.py
├── requirements.txt
└── README.md
```

---

## 📚 References

1. Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (pix2pix)
2. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
3. Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
4. CelebA-HQ Dataset: https://github.com/tkarras/progressive_growing_of_gans

---

_Last updated: January 2026_
