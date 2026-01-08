"""
Plot Training Metrics from CSV Log
===================================

This script:
1. Reads train_log.csv file
2. Plots all metrics using Seaborn
3. Finds the best checkpoint based on metrics
4. Saves statistics and plots

Usage:
    python plot_metrics.py
    python plot_metrics.py --log_file results/train_log.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from datetime import datetime


def load_training_log(log_file):
    """Load training log from CSV file."""
    df = pd.read_csv(log_file)
    print(f"✓ Loaded {len(df)} epochs from {log_file}")
    return df


def find_best_checkpoint(df):
    """
    Find the best checkpoint based on multiple metrics.
    Best = highest SSIM and PSNR, lowest losses
    """
    # Normalize metrics for comparison (higher is better for SSIM/PSNR, lower for losses)
    df_norm = df.copy()
    
    # For SSIM and PSNR: higher is better (normalize to 0-1, higher = better)
    df_norm['SSIM_score'] = (df['SSIM'] - df['SSIM'].min()) / (df['SSIM'].max() - df['SSIM'].min())
    df_norm['PSNR_score'] = (df['PSNR'] - df['PSNR'].min()) / (df['PSNR'].max() - df['PSNR'].min())
    
    # For losses: lower is better (normalize to 0-1, then invert)
    df_norm['G_Loss_score'] = 1 - (df['G_Loss'] - df['G_Loss'].min()) / (df['G_Loss'].max() - df['G_Loss'].min())
    df_norm['L1_Loss_score'] = 1 - (df['L1_Loss'] - df['L1_Loss'].min()) / (df['L1_Loss'].max() - df['L1_Loss'].min())
    df_norm['VGG_Loss_score'] = 1 - (df['VGG_Loss'] - df['VGG_Loss'].min()) / (df['VGG_Loss'].max() - df['VGG_Loss'].min())
    
    # Combined score (weighted average)
    df_norm['combined_score'] = (
        0.3 * df_norm['SSIM_score'] +
        0.3 * df_norm['PSNR_score'] +
        0.15 * df_norm['G_Loss_score'] +
        0.15 * df_norm['L1_Loss_score'] +
        0.1 * df_norm['VGG_Loss_score']
    )
    
    best_idx = df_norm['combined_score'].idxmax()
    best_epoch = df.loc[best_idx, 'Epoch']
    
    return int(best_epoch), df.loc[best_idx]


def plot_metrics(df, output_dir):
    """Plot all training metrics using Seaborn."""
    
    # Set style
    sns.set_theme(style="darkgrid", palette="husl")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find best epoch for annotation
    best_epoch, best_metrics = find_best_checkpoint(df)
    
    # ==================== Plot 1: All Losses ====================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # D_Loss
    ax = axes[0, 0]
    sns.lineplot(data=df, x='Epoch', y='D_Loss', ax=ax, color='#e74c3c', linewidth=2)
    ax.set_title('Discriminator Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('D_Loss')
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.legend()
    
    # G_Loss
    ax = axes[0, 1]
    sns.lineplot(data=df, x='Epoch', y='G_Loss', ax=ax, color='#3498db', linewidth=2)
    ax.set_title('Generator Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('G_Loss')
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.legend()
    
    # L1_Loss
    ax = axes[1, 0]
    sns.lineplot(data=df, x='Epoch', y='L1_Loss', ax=ax, color='#9b59b6', linewidth=2)
    ax.set_title('L1 Loss (Reconstruction)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1_Loss')
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.legend()
    
    # VGG_Loss
    ax = axes[1, 1]
    sns.lineplot(data=df, x='Epoch', y='VGG_Loss', ax=ax, color='#e67e22', linewidth=2)
    ax.set_title('VGG Perceptual Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VGG_Loss')
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.legend()
    
    plt.suptitle('Training Losses Over Epochs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses_plot.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: losses_plot.png")
    
    # ==================== Plot 2: Quality Metrics ====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SSIM
    ax = axes[0]
    sns.lineplot(data=df, x='Epoch', y='SSIM', ax=ax, color='#27ae60', linewidth=2.5)
    ax.fill_between(df['Epoch'], df['SSIM'], alpha=0.3, color='#27ae60')
    ax.set_title('SSIM (Structural Similarity)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.axhline(y=best_metrics['SSIM'], color='red', linestyle=':', alpha=0.5)
    ax.annotate(f'{best_metrics["SSIM"]:.4f}', xy=(best_epoch, best_metrics['SSIM']), 
                xytext=(best_epoch+2, best_metrics['SSIM']+0.01), fontsize=10, color='red')
    ax.legend()
    
    # PSNR
    ax = axes[1]
    sns.lineplot(data=df, x='Epoch', y='PSNR', ax=ax, color='#2980b9', linewidth=2.5)
    ax.fill_between(df['Epoch'], df['PSNR'], alpha=0.3, color='#2980b9')
    ax.set_title('PSNR (Peak Signal-to-Noise Ratio)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
    ax.axhline(y=best_metrics['PSNR'], color='red', linestyle=':', alpha=0.5)
    ax.annotate(f'{best_metrics["PSNR"]:.2f} dB', xy=(best_epoch, best_metrics['PSNR']), 
                xytext=(best_epoch+2, best_metrics['PSNR']+0.3), fontsize=10, color='red')
    ax.legend()
    
    plt.suptitle('Image Quality Metrics Over Epochs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_metrics_plot.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: quality_metrics_plot.png")
    
    # ==================== Plot 3: Combined Overview ====================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics_config = [
        ('D_Loss', 'Discriminator Loss', '#e74c3c'),
        ('G_Loss', 'Generator Loss', '#3498db'),
        ('L1_Loss', 'L1 Loss', '#9b59b6'),
        ('VGG_Loss', 'VGG Loss', '#e67e22'),
        ('SSIM', 'SSIM', '#27ae60'),
        ('PSNR', 'PSNR (dB)', '#2980b9'),
    ]
    
    for idx, (col, title, color) in enumerate(metrics_config):
        ax = axes[idx // 3, idx % 3]
        sns.lineplot(data=df, x='Epoch', y=col, ax=ax, color=color, linewidth=2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.6)
        
        # Mark best point
        best_val = best_metrics[col]
        ax.scatter([best_epoch], [best_val], color='red', s=100, zorder=5, marker='*')
    
    plt.suptitle(f'Training Overview - Best Checkpoint: Epoch {best_epoch}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: training_overview.png")
    
    # ==================== Plot 4: Correlation Heatmap ====================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_cols = ['D_Loss', 'G_Loss', 'L1_Loss', 'VGG_Loss', 'SSIM', 'PSNR']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, 
                fmt='.3f', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: correlation_heatmap.png")
    
    plt.show()
    
    return best_epoch, best_metrics


def save_statistics(df, best_epoch, best_metrics, output_dir):
    """Save training statistics to text file."""
    
    stats_file = os.path.join(output_dir, 'training_statistics.txt')
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("📊 TRAINING STATISTICS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {len(df)}\n")
        f.write("\n")
        
        # Best Checkpoint
        f.write("-" * 60 + "\n")
        f.write(f"🏆 BEST CHECKPOINT: Epoch {best_epoch}\n")
        f.write("-" * 60 + "\n")
        f.write(f"  D_Loss:    {best_metrics['D_Loss']:.6f}\n")
        f.write(f"  G_Loss:    {best_metrics['G_Loss']:.6f}\n")
        f.write(f"  L1_Loss:   {best_metrics['L1_Loss']:.6f}\n")
        f.write(f"  VGG_Loss:  {best_metrics['VGG_Loss']:.6f}\n")
        f.write(f"  SSIM:      {best_metrics['SSIM']:.6f}\n")
        f.write(f"  PSNR:      {best_metrics['PSNR']:.2f} dB\n")
        f.write("\n")
        
        # Final Epoch Statistics
        final_metrics = df.iloc[-1]
        f.write("-" * 60 + "\n")
        f.write(f"📈 FINAL EPOCH: Epoch {int(final_metrics['Epoch'])}\n")
        f.write("-" * 60 + "\n")
        f.write(f"  D_Loss:    {final_metrics['D_Loss']:.6f}\n")
        f.write(f"  G_Loss:    {final_metrics['G_Loss']:.6f}\n")
        f.write(f"  L1_Loss:   {final_metrics['L1_Loss']:.6f}\n")
        f.write(f"  VGG_Loss:  {final_metrics['VGG_Loss']:.6f}\n")
        f.write(f"  SSIM:      {final_metrics['SSIM']:.6f}\n")
        f.write(f"  PSNR:      {final_metrics['PSNR']:.2f} dB\n")
        f.write("\n")
        
        # Overall Statistics
        f.write("-" * 60 + "\n")
        f.write("📊 OVERALL STATISTICS\n")
        f.write("-" * 60 + "\n")
        
        for col in ['D_Loss', 'G_Loss', 'L1_Loss', 'VGG_Loss', 'SSIM', 'PSNR']:
            f.write(f"\n{col}:\n")
            f.write(f"  Min:    {df[col].min():.6f}\n")
            f.write(f"  Max:    {df[col].max():.6f}\n")
            f.write(f"  Mean:   {df[col].mean():.6f}\n")
            f.write(f"  Std:    {df[col].std():.6f}\n")
        
        # Improvement from first to best
        f.write("\n" + "-" * 60 + "\n")
        f.write("📈 IMPROVEMENT (First → Best Epoch)\n")
        f.write("-" * 60 + "\n")
        
        first_metrics = df.iloc[0]
        ssim_improve = (best_metrics['SSIM'] - first_metrics['SSIM']) / first_metrics['SSIM'] * 100
        psnr_improve = (best_metrics['PSNR'] - first_metrics['PSNR']) / first_metrics['PSNR'] * 100
        g_loss_improve = (first_metrics['G_Loss'] - best_metrics['G_Loss']) / first_metrics['G_Loss'] * 100
        
        f.write(f"  SSIM:    +{ssim_improve:.2f}%\n")
        f.write(f"  PSNR:    +{psnr_improve:.2f}%\n")
        f.write(f"  G_Loss:  -{g_loss_improve:.2f}% (reduced)\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Saved: {stats_file}")
    
    # Also print to console
    with open(stats_file, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from CSV log")
    parser.add_argument("--log_file", type=str, 
                        default="results/train_log.csv",
                        help="Path to training log CSV file")
    parser.add_argument("--output_dir", type=str,
                        default="results/plots",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, args.log_file) if not os.path.isabs(args.log_file) else args.log_file
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    print("=" * 60)
    print("📊 TRAINING METRICS VISUALIZATION")
    print("=" * 60)
    
    # Load data
    df = load_training_log(log_file)
    
    # Plot metrics
    best_epoch, best_metrics = plot_metrics(df, output_dir)
    
    # Save statistics
    save_statistics(df, best_epoch, best_metrics, output_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ All plots saved to: {output_dir}")
    print(f"🏆 Best checkpoint: gen_{best_epoch}.pth.tar")
    print("=" * 60)


if __name__ == "__main__":
    main()
