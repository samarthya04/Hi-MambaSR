"""
Hi-MambaSR Publication Figure Generator
=======================================
Generates all remaining publication-quality figures after training.

Usage:
    # Generate all figures from a trained checkpoint:
    python generate_figures.py model.load_model=models/checkpoints/best.ckpt

    # Generate specific figures only:
    python generate_figures.py model.load_model=models/checkpoints/best.ckpt \
        figures=[visual_comparison,diffusion_trajectory,frequency_analysis,vram_profile]

    # Generate training curves from W&B (no checkpoint needed):
    python generate_figures.py figures=[training_curves] wandb.run_id=<YOUR_RUN_ID>

Figures Generated:
    Fig 2: Qualitative visual comparison grid (visual_comparison)
    Fig 6: Training curves from W&B logs (training_curves)
    Fig 8: Diffusion denoising trajectory (diffusion_trajectory)
    Fig 9: Frequency / edge analysis (frequency_analysis)
    Extra: VRAM profiling measurement (vram_profile)
"""

import os
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.size'] = 10

from scripts.data_loader import train_val_test_loader
from scripts.model_config import model_selection

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

# Resolved dynamically in main() to handle Hydra CWD changes
FIGURE_DIR = None

def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] tensor (B,C,H,W) to [0,1] numpy (B,H,W,C)."""
    return np.clip((x.detach().cpu().float().numpy().transpose(0, 2, 3, 1) + 1) / 2, 0, 1)

def rgb_to_ycbcr_y(img_np: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] numpy (H,W,3) to Y channel."""
    return 16./255. + (65.481/255. * img_np[..., 0] + 128.553/255. * img_np[..., 1] + 24.966/255. * img_np[..., 2])

def compute_metrics(sr_np, hr_np, border=4):
    """Compute Y-channel PSNR and SSIM for a single image pair."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    hr_y = rgb_to_ycbcr_y(hr_np)
    sr_y = rgb_to_ycbcr_y(sr_np)
    if border > 0:
        hr_y = hr_y[border:-border, border:-border]
        sr_y = sr_y[border:-border, border:-border]
    psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0)
    ssim = structural_similarity(hr_y, sr_y, data_range=1.0)
    return psnr, ssim

def save_figure(fig, name: str, dpi=300):
    """Save figure to the figures directory."""
    global FIGURE_DIR
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.1)
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 2: Qualitative Visual Comparison
# ──────────────────────────────────────────────────────────────────────────────

def generate_visual_comparison(model, test_loader, device, num_samples=4):
    """
    Generate a side-by-side visual comparison grid.
    Columns: LR Bicubic ↑4× | Hi-MambaSR (Ours) | Ground Truth HR
    Each row includes a zoomed-in crop patch.
    """
    print("\n📊 Generating Fig 2: Qualitative Visual Comparison...")
    model.eval()
    
    samples_lr, samples_sr, samples_hr = [], [], []
    metrics_per_sample = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_samples:
                break
            lr_img = batch["lr"].to(device).float()
            hr_img = batch["hr"].to(device).float()
            sr_img = model(lr_img).float()
            
            samples_lr.append(lr_img[0])
            samples_sr.append(sr_img[0])
            samples_hr.append(hr_img[0])
            
            sr_np = tensor_to_numpy(sr_img)[0]
            hr_np = tensor_to_numpy(hr_img)[0]
            psnr, ssim = compute_metrics(sr_np, hr_np)
            metrics_per_sample.append((psnr, ssim))
    
    n = len(samples_lr)
    
    # Main comparison grid: 2 rows per sample (full image + crop)
    fig, axes = plt.subplots(n * 2, 3, figsize=(12, 4 * n), dpi=150)
    fig.patch.set_facecolor('white')
    
    col_titles = ["Bicubic ↑4×", "Hi-MambaSR (Ours)", "Ground Truth"]
    
    for i in range(n):
        lr_np = tensor_to_numpy(samples_lr[i].unsqueeze(0))[0]
        sr_np = tensor_to_numpy(samples_sr[i].unsqueeze(0))[0]
        hr_np = tensor_to_numpy(samples_hr[i].unsqueeze(0))[0]
        
        # Bicubic upscale of LR for display
        lr_up = tensor_to_numpy(
            F.interpolate(samples_lr[i].unsqueeze(0), size=samples_hr[i].shape[-2:], 
                         mode='bicubic', align_corners=False)
        )[0]
        
        images = [lr_up, sr_np, hr_np]
        psnr, ssim = metrics_per_sample[i]
        
        # Define crop region (center 25% of the image)
        h, w = hr_np.shape[:2]
        ch, cw = h // 4, w // 4
        cy, cx = h // 2 - ch // 2, w // 2 - cw // 2
        
        for j, img in enumerate(images):
            # Full image row
            ax = axes[i * 2, j] if n > 1 else axes[0, j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title(col_titles[j], fontsize=13, fontweight='bold', pad=10)
            
            # Draw crop rectangle on full image
            rect = plt.Rectangle((cx, cy), cw, ch, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            # Cropped zoom row
            ax_crop = axes[i * 2 + 1, j] if n > 1 else axes[1, j]
            ax_crop.imshow(img[cy:cy+ch, cx:cx+cw])
            ax_crop.axis('off')
            ax_crop.set_title(f"Zoomed Crop", fontsize=9, color='red') if j == 0 and i == 0 else None
        
        # Add metrics annotation
        ax_sr_crop = axes[i * 2 + 1, 1] if n > 1 else axes[1, 1]
        ax_sr_crop.text(0.02, 0.02, f"PSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}", 
                       transform=ax_sr_crop.transAxes, fontsize=8, color='white',
                       fontweight='bold', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    fig.suptitle("Qualitative Comparison (4× Super-Resolution)", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, "fig2_visual_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 6: Training Curves
# ──────────────────────────────────────────────────────────────────────────────

def generate_training_curves(cfg):
    """
    Generate training curves from W&B logs or local CSV.
    Plots: Generator Loss, PSNR, SSIM, LPIPS vs. epoch.
    """
    print("\n📊 Generating Fig 6: Training Curves...")
    
    # Try to load from W&B
    try:
        import wandb
        api = wandb.Api()
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}"
        
        # Get the latest run if no specific run_id provided
        run_id = OmegaConf.select(cfg, "wandb.run_id", default=None)
        if run_id:
            run = api.run(f"{run_path}/{run_id}")
        else:
            runs = api.runs(run_path, order="-created_at")
            if not runs:
                print("  ⚠ No W&B runs found. Skipping training curves.")
                return
            run = runs[0]
            print(f"  Using latest run: {run.name} ({run.id})")
        
        history = run.history(samples=10000)
        
        if history.empty:
            print("  ⚠ No history data in W&B run. Skipping.")
            return
            
    except Exception as e:
        print(f"  ⚠ Could not load W&B data: {e}")
        print("  Trying local CSV fallback...")
        
        csv_path = Path("evaluation_results/hi_mambasr_benchmarks.csv")
        if csv_path.exists():
            import pandas as pd
            history = pd.read_csv(csv_path)
        else:
            print("  ⚠ No local CSV found either. Skipping training curves.")
            return
    
    # Define plot configurations
    plot_configs = [
        {
            "title": "Generator Loss",
            "keys": ["train/g_loss"],
            "ylabel": "Loss",
            "colors": ["#2196F3"],
            "labels": ["G Loss"],
        },
        {
            "title": "Validation PSNR",
            "keys": ["val/PSNR"],
            "ylabel": "PSNR (dB) ↑",
            "colors": ["#4CAF50"],
            "labels": ["PSNR"],
        },
        {
            "title": "Validation LPIPS", 
            "keys": ["val/LPIPS"],
            "ylabel": "LPIPS ↓",
            "colors": ["#FF5722"],
            "labels": ["LPIPS"],
        },
        {
            "title": "Discriminator Dynamics",
            "keys": ["train/d_loss", "train/ema_s"],
            "ylabel": "Value",
            "colors": ["#9C27B0", "#FF9800"],
            "labels": ["D Loss", "Noise Step s"],
        },
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, config in enumerate(plot_configs):
        ax = axes[idx]
        ax.set_facecolor('#FAFAFA')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_title(config["title"], fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(config["ylabel"], fontsize=10)
        
        for key, color, label in zip(config["keys"], config["colors"], config["labels"]):
            if key in history.columns:
                data = history[key].dropna()
                if len(data) > 0:
                    # Smooth with rolling average for cleaner curves
                    if len(data) > 50:
                        smoothed = data.rolling(window=max(1, len(data) // 50), min_periods=1).mean()
                        ax.plot(smoothed.index, smoothed.values, color=color, label=f"{label} (smoothed)", 
                               linewidth=1.5, alpha=0.9)
                        ax.plot(data.index, data.values, color=color, alpha=0.15, linewidth=0.5)
                    else:
                        ax.plot(data.index, data.values, color=color, label=label, linewidth=1.5)
                    ax.legend(fontsize=9, framealpha=0.8)
            else:
                ax.text(0.5, 0.5, f"'{key}' not found in logs", transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray', style='italic')
    
    fig.suptitle("Hi-MambaSR Training Dynamics", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "fig6_training_curves")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 8: Diffusion Denoising Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def generate_diffusion_trajectory(model, test_loader, device, num_vis_steps=8):
    """
    Visualize the latent diffusion denoising process from t=T to t=0.
    Shows progressive refinement through the reverse trajectory.
    """
    print("\n📊 Generating Fig 8: Diffusion Denoising Trajectory...")
    model.eval()
    
    # Get a single test sample
    batch = next(iter(test_loader))
    lr_img = batch["lr"][:1].to(device).float()
    hr_img = batch["hr"][:1].to(device).float()
    
    gen = model.ema_generator if model.ema_generator is not None else model.generator
    diffusion = model.diffusion
    
    # Store original settings
    orig_timesteps = diffusion.timesteps
    orig_posterior = diffusion.posterior_type
    
    # Use DDIM with moderate steps for clean trajectory
    vis_timesteps = 50
    diffusion.set_timesteps(vis_timesteps)
    diffusion.set_posterior_type('ddim')
    
    with torch.no_grad():
        # Encode LR to latent
        posterior = model.ae.encode(lr_img).latent_dist
        lr_lat = posterior.mode() * model.ae.config.scaling_factor
        
        # Initialize from noise
        x_t = torch.randn_like(lr_lat).to(device)
        ab_cache = diffusion.alpha_bars_torch.to(device)
        
        # Capture snapshots at evenly-spaced intervals
        snapshot_steps = set(np.linspace(vis_timesteps - 1, 0, num_vis_steps, dtype=int).tolist())
        snapshots = []  # (timestep_index, decoded_image)
        
        for i in reversed(range(vis_timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            pred_x_0 = gen(lr_lat, x_t, ab_cache[t])
            
            if i in snapshot_steps:
                # Decode prediction to pixel space for visualization
                decoded = model.ae.decode(
                    pred_x_0.to(torch.float32) / model.ae.config.scaling_factor
                ).sample
                lr_up = F.interpolate(lr_img, size=decoded.shape[-2:], mode='bicubic', align_corners=False)
                sr_snap = torch.clamp(decoded + lr_up, -1, 1)
                snapshots.append((i, tensor_to_numpy(sr_snap)[0]))
            
            if i > 0:
                x_t = diffusion.ddim_posterior(x_t, pred_x_0, t)
            else:
                x_t = pred_x_0
        
        # Final output
        final_decoded = model.ae.decode(
            x_t.to(torch.float32) / model.ae.config.scaling_factor
        ).sample
        lr_up = F.interpolate(lr_img, size=final_decoded.shape[-2:], mode='bicubic', align_corners=False)
        final_sr = torch.clamp(final_decoded + lr_up, -1, 1)
    
    # Restore settings
    diffusion.set_timesteps(orig_timesteps)
    diffusion.set_posterior_type(orig_posterior)
    
    # Sort snapshots by timestep (high to low)
    snapshots.sort(key=lambda x: x[0], reverse=True)
    
    # Build figure: LR | snapshots... | Final SR | HR
    lr_np = tensor_to_numpy(lr_img)[0]
    lr_up_np = tensor_to_numpy(
        F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
    )[0]
    hr_np = tensor_to_numpy(hr_img)[0]
    final_np = tensor_to_numpy(final_sr)[0]
    
    n_cols = 2 + len(snapshots) + 1  # LR + snapshots + final + HR
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5), dpi=150)
    fig.patch.set_facecolor('white')
    
    # LR Bicubic
    axes[0].imshow(lr_up_np)
    axes[0].set_title("LR Bicubic ↑4×", fontsize=9, fontweight='bold')
    axes[0].axis('off')
    
    # Denoising snapshots
    for idx, (t_step, snap_img) in enumerate(snapshots):
        axes[idx + 1].imshow(snap_img)
        noise_pct = int(100 * t_step / vis_timesteps)
        axes[idx + 1].set_title(f"t={t_step} ({noise_pct}%)", fontsize=9)
        axes[idx + 1].axis('off')
    
    # Final SR
    axes[-2].imshow(final_np)
    axes[-2].set_title("Final SR (t=0)", fontsize=9, fontweight='bold', color='green')
    axes[-2].axis('off')
    
    # Ground Truth
    axes[-1].imshow(hr_np)
    axes[-1].set_title("Ground Truth", fontsize=9, fontweight='bold', color='blue')
    axes[-1].axis('off')
    
    # Add directional arrow below
    fig.text(0.5, -0.02, "← Reverse Diffusion (Denoising) →", ha='center', fontsize=12, 
             fontweight='bold', color='#555555')
    
    fig.suptitle("Latent Diffusion Denoising Trajectory (DDIM, 50 steps)", 
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    save_figure(fig, "fig8_diffusion_trajectory")


# ──────────────────────────────────────────────────────────────────────────────
# Fig 9: Frequency / Edge Analysis
# ──────────────────────────────────────────────────────────────────────────────

def generate_frequency_analysis(model, test_loader, device):
    """
    Show that Hi-MambaSR recovers high-frequency details via:
    1. FFT magnitude spectrum comparison
    2. Sobel edge map comparison
    """
    print("\n📊 Generating Fig 9: Frequency & Edge Analysis...")
    model.eval()
    
    batch = next(iter(test_loader))
    lr_img = batch["lr"][:1].to(device).float()
    hr_img = batch["hr"][:1].to(device).float()
    
    with torch.no_grad():
        sr_img = model(lr_img).float()
    
    lr_up_np = tensor_to_numpy(
        F.interpolate(lr_img, size=hr_img.shape[-2:], mode='bicubic', align_corners=False)
    )[0]
    sr_np = tensor_to_numpy(sr_img)[0]
    hr_np = tensor_to_numpy(hr_img)[0]
    
    def compute_fft_magnitude(img_np):
        """Compute log FFT magnitude of grayscale image."""
        gray = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        return magnitude / magnitude.max()  # Normalize to [0, 1]
    
    def compute_sobel_edges(img_np):
        """Compute Sobel edge magnitude on grayscale."""
        gray = 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]
        gray_t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        padded = F.pad(gray_t, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(padded, kx)
        gy = F.conv2d(padded, ky)
        magnitude = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return magnitude.squeeze().numpy()
    
    images = [lr_up_np, sr_np, hr_np]
    titles = ["Bicubic ↑4×", "Hi-MambaSR (Ours)", "Ground Truth"]
    
    fig = plt.figure(figsize=(15, 12), dpi=150)
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.2)
    
    for col, (img, title) in enumerate(zip(images, titles)):
        # Row 1: Original image
        ax1 = fig.add_subplot(gs[0, col])
        ax1.imshow(img)
        ax1.set_title(title, fontsize=12, fontweight='bold', pad=8)
        ax1.axis('off')
        
        # Row 2: FFT magnitude spectrum
        fft_mag = compute_fft_magnitude(img)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(fft_mag, cmap='inferno')
        ax2.set_title("FFT Magnitude Spectrum", fontsize=10, pad=6)
        ax2.axis('off')
        
        # Row 3: Sobel edge map
        edges = compute_sobel_edges(img)
        ax3 = fig.add_subplot(gs[2, col])
        ax3.imshow(edges, cmap='gray')
        ax3.set_title("Sobel Edge Map", fontsize=10, pad=6)
        ax3.axis('off')
    
    # Row labels on the left
    fig.text(0.02, 0.83, "RGB Image", fontsize=11, rotation=90, va='center', fontweight='bold', color='#333')
    fig.text(0.02, 0.50, "Frequency\nDomain", fontsize=11, rotation=90, va='center', fontweight='bold', color='#333')
    fig.text(0.02, 0.17, "Edge\nDetection", fontsize=11, rotation=90, va='center', fontweight='bold', color='#333')
    
    fig.suptitle("Frequency & Edge Analysis — High-Frequency Detail Recovery", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    save_figure(fig, "fig9_frequency_analysis")


# ──────────────────────────────────────────────────────────────────────────────
# VRAM Profiler
# ──────────────────────────────────────────────────────────────────────────────

def profile_vram(model, test_loader, cfg, device):
    """
    Measure actual peak VRAM during a training step and inference.
    Prints results and saves to a text report.
    """
    print("\n📊 Profiling VRAM Usage...")
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available. Skipping VRAM profiling.")
        return
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    results = {}
    
    # --- Inference VRAM ---
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        batch = next(iter(test_loader))
        lr_img = batch["lr"][:1].to(device).float()
        sr_img = model(lr_img)
    
    results["inference_peak_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    results["inference_peak_gb"] = results["inference_peak_mb"] / 1024
    torch.cuda.empty_cache()
    
    # --- Training VRAM (single step simulation) ---
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Initialize EMA if needed
    if model.ema_generator is None:
        import copy
        model.ema_generator = copy.deepcopy(model._raw_generator)
        model.ema_generator.eval()
        for p in model.ema_generator.parameters():
            p.requires_grad = False
    
    batch = next(iter(test_loader))
    lr_img = batch["lr"][:cfg.dataset.batch_size].to(device).float()
    hr_img = batch["hr"][:cfg.dataset.batch_size].to(device).float()
    
    scale = model.ae.config.scaling_factor
    with torch.no_grad():
        lr_lat = model.ae.encode(lr_img).latent_dist.mode().detach() * scale
        x0_lat = model.ae.encode(hr_img).latent_dist.mode().detach() * scale
    
    t = torch.randint(0, model.diffusion.timesteps, (x0_lat.shape[0],), device=device)
    x_t = model.diffusion.forward(x0_lat, t)
    alfa_bars = model.diffusion.alpha_bars_torch.to(device)[t]
    
    x_gen = model.generator(lr_lat, x_t, alfa_bars)
    loss = F.l1_loss(x_gen, x0_lat)
    loss.backward()
    
    results["training_peak_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Clean up gradients from profiling
    model.zero_grad(set_to_none=True)
    results["training_peak_gb"] = results["training_peak_mb"] / 1024
    
    # Model parameter counts
    gen_params = sum(p.numel() for p in model.generator.parameters()) / 1e6
    disc_params = sum(p.numel() for p in model.discriminator.parameters()) / 1e6 if model.discriminator else 0
    
    results["generator_params_M"] = gen_params
    results["discriminator_params_M"] = disc_params
    results["total_trainable_M"] = gen_params + disc_params
    
    # Save report
    FIGURE_DIR.mkdir(exist_ok=True)
    report_path = FIGURE_DIR / "vram_profile_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Hi-MambaSR VRAM & Parameter Profile\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(device)}\n")
        f.write(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB\n\n")
        f.write(f"--- Inference (batch_size=1) ---\n")
        f.write(f"Peak VRAM: {results['inference_peak_mb']:.1f} MB ({results['inference_peak_gb']:.2f} GB)\n\n")
        f.write(f"--- Training (batch_size={cfg.dataset.batch_size}) ---\n")
        f.write(f"Peak VRAM: {results['training_peak_mb']:.1f} MB ({results['training_peak_gb']:.2f} GB)\n\n")
        f.write(f"--- Model Complexity ---\n")
        f.write(f"Generator Parameters: {gen_params:.2f}M\n")
        f.write(f"Discriminator Parameters: {disc_params:.2f}M\n")
        f.write(f"Total Trainable: {gen_params + disc_params:.2f}M\n")
    
    print(f"  ✓ VRAM Profile Report saved: {report_path}")
    print(f"  📌 Training Peak: {results['training_peak_gb']:.2f} GB")
    print(f"  📌 Inference Peak: {results['inference_peak_gb']:.2f} GB")
    print(f"  📌 Generator: {gen_params:.2f}M params")
    
    model.eval()
    torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

ALL_FIGURES = [
    "visual_comparison",
    "training_curves", 
    "diffusion_trajectory",
    "frequency_analysis",
    "vram_profile",
]

@hydra.main(version_base=None, config_path="conf", config_name="config_mamba")
def main(cfg: DictConfig) -> None:
    """Generate publication figures for Hi-MambaSR."""
    
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    
    # Resolve FIGURE_DIR relative to the original project directory (not Hydra's CWD)
    global FIGURE_DIR
    FIGURE_DIR = Path(get_original_cwd()) / "figures"
    
    # Parse which figures to generate
    figures = OmegaConf.select(cfg, "figures", default=ALL_FIGURES)
    if isinstance(figures, str):
        figures = [figures]
    figures = list(figures)
    
    print("=" * 60)
    print("Hi-MambaSR Publication Figure Generator")
    print("=" * 60)
    print(f"Figures to generate: {figures}")
    print(f"Output directory: {FIGURE_DIR.absolute()}")
    
    # Training curves can be generated without a model checkpoint
    if figures == ["training_curves"]:
        generate_training_curves(cfg)
        print("\n✅ Done!")
        return
    
    # All other figures need a trained model
    if cfg.model.load_model is None:
        print("\n⚠ No checkpoint specified. Use: model.load_model=<path>")
        print("  Only 'training_curves' can be generated without a checkpoint.")
        
        if "training_curves" in figures:
            generate_training_curves(cfg)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    print(f"\nLoading Hi-MambaSR from: {cfg.model.load_model}")
    model = model_selection(cfg=cfg, device=device)
    model.to(device)
    model.eval()
    
    # Load test data
    _, _, test_loader = train_val_test_loader(cfg=cfg)
    
    print(f"Model loaded on {device}. Generating figures...\n")
    
    # Generate requested figures
    if "visual_comparison" in figures:
        generate_visual_comparison(model, test_loader, device, num_samples=4)
    
    if "training_curves" in figures:
        generate_training_curves(cfg)
    
    if "diffusion_trajectory" in figures:
        generate_diffusion_trajectory(model, test_loader, device, num_vis_steps=8)
    
    if "frequency_analysis" in figures:
        generate_frequency_analysis(model, test_loader, device)
    
    if "vram_profile" in figures:
        profile_vram(model, test_loader, cfg, device)
    
    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print(f"📁 Output: {FIGURE_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
