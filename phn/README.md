remotes/origin/cursor/face-inpainting-convergence-issue-6f92

# GANsformer Face Inpainting

A PyTorch implementation of GANsformer for face inpainting, combining the power of CNNs and Transformers to achieve high-quality image completion.

## Overview

This project implements a GANsformer-based architecture for face inpainting. The model uses a U-Net style generator with GANsformer blocks in the bottleneck, enabling global context modeling through bipartite attention mechanisms. The discriminator also incorporates bipartite attention for improved global reasoning.

**Key Innovation**: CNN-GAN + Transformer = GANsformer

## Features

- **Bipartite Attention**: Bidirectional communication between pixel features and latent variables
- **U-Net Generator**: Encoder-bottleneck-decoder architecture with GANsformer blocks
- **Global Context Modeling**: Transformer-based attention for long-range dependencies
- **R1 Regularization**: Gradient penalty for stable training
- **Weights & Biases Integration**: Comprehensive training monitoring and visualization
- **Checkpoint Management**: Save and resume training from checkpoints

## Architecture

### Generator

- **Encoder**: Progressive downsampling (256×256 → 32×32) using strided convolutions
- **Bottleneck**: Stacked GANsformer blocks with bipartite attention for global context
- **Decoder**: Progressive upsampling (32×32 → 256×256) using transposed convolutions

### Discriminator

- **Feature Extractor**: Strided convolutions for spatial downsampling
- **Bipartite Attention**: Global context aggregation
- **Classifier**: Real/fake prediction with global pooling

### Bipartite Attention

The core mechanism enabling bidirectional communication:

1. **Aggregation Phase**: Latent variables gather information from pixel features
2. **Broadcast Phase**: Pixel features receive global context from updated latents

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- See `requirements.txt` for full dependencies

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd dl
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install Weights & Biases for experiment tracking:

```bash
pip install wandb
wandb login
```

## Dataset Setup

The model expects a dataset with the following structure:

```
dataset/celeba_hq_256/
├── extracted/          # Masked images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ground_truth/      # Ground truth images
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

**Important**:

- Masked and ground truth images must have matching filenames
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- Images will be automatically resized to 256×256 during training

## Usage

### Basic Training

Train with default configuration:

```bash
python main.py
```

### Custom Training Parameters

```bash
# Train for 10 epochs with batch size 8
python main.py --epochs 10 --batch-size 8

# Use CPU instead of CUDA
python main.py --device cpu

# Specify custom dataset path
python main.py --data-root path/to/your/dataset

# Adjust learning rate
python main.py --lr 0.001
```

### Resume Training

Resume from a checkpoint:

```bash
python main.py --resume checkpoints/checkpoint_1000.pt
```

### Weights & Biases Logging

Enable/disable W&B logging:

```bash
# Enable (default)
python main.py --wandb

# Disable
python main.py --no-wandb

# Custom W&B project
python main.py --wandb-project my-project --wandb-entity my-username
```

### Full Command-Line Options

```bash
python main.py --help
```

Available options:

- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--data-root`: Root directory for dataset
- `--device`: Device to use (cuda/cpu)
- `--workers`: Number of data loading workers
- `--resume`: Path to checkpoint to resume from
- `--wandb` / `--no-wandb`: Enable/disable wandb logging
- `--wandb-project`: Wandb project name
- `--wandb-entity`: Wandb entity (username/team)
- `--wandb-name`: Wandb run name

## Configuration

Default configuration can be modified in `src/config/config.py`. Key parameters:

### Model Architecture

- `latent_num`: Number of latent variables (default: 32)
- `latent_dim`: Feature dimension (default: 512)
- `num_heads`: Multi-head attention heads (default: 8)
- `num_gansformer_blocks`: Number of GANsformer blocks in bottleneck (default: 2)

### Training Hyperparameters

- `learning_rate`: Learning rate (default: 0.002)
- `num_epochs`: Number of epochs (default: 3)
- `batch_size`: Batch size (default: 16)
- `l1_weight`: L1 reconstruction loss weight (default: 100.0)
- `r1_gamma`: R1 gradient penalty weight (default: 10.0)
- `d_reg_interval`: Lazy regularization interval (default: 16)

### Data Settings

- `img_size`: Image size (default: 256)
- `data_root`: Dataset root directory (default: `dataset/celeba_hq_256`)

## Project Structure

```
dl/
├── main.py                 # Main training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── paper/                 # Reference papers
│   ├── attention.pdf
│   ├── gan.pdf
│   └── gansformer.pdf
└── src/
    ├── config/
    │   ├── __init__.py
    │   └── config.py      # Configuration settings
    ├── dataloader/
    │   ├── __init__.py
    │   └── loader.py      # Dataset and DataLoader
    ├── models/
    │   ├── __init__.py
    │   ├── bipartite.py   # Bipartite attention mechanism
    │   ├── generator.py   # Generator architecture
    │   └── discriminator.py  # Discriminator architecture
    └── training/
        ├── __init__.py
        └── trainer.py     # Training loop and utilities
```

## Training Details

### Loss Functions

**Generator Loss**:

- Adversarial loss: Non-saturating GAN loss
- L1 reconstruction loss: Pixel-wise reconstruction

**Discriminator Loss**:

- Real/fake classification loss
- R1 gradient penalty (applied every `d_reg_interval` steps)

### Training Strategy

- **Lazy Regularization**: R1 penalty applied every N steps for efficiency
- **Checkpointing**: Automatic checkpoint saving every `save_interval` steps
- **Logging**: Console and W&B logging at specified intervals

### Monitoring

Training metrics logged:

- Generator loss (adversarial + L1)
- Discriminator loss
- R1 regularization loss
- Sample image comparisons (masked | generated | ground truth)

## References

- **GANsformer**: Generative Adversarial Transformers
- **Attention Is All You Need**: Transformer architecture
- **Generative Adversarial Networks**: GAN fundamentals

See `paper/` directory for reference papers.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

This implementation is based on the GANsformer architecture, combining CNNs and Transformers for improved generative modeling
