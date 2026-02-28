# Hi-MambaSR

### Hierarchical State-Space Refinement for Latent Diffusion Super-Resolution

<div align="center">
  <p><em>A hybrid framework unifying Latent Denoising Diffusion, Relativistic GANs, Swin Transformers, and Mamba State-Space Models for high-fidelity single-image super-resolution — engineered to train end-to-end on 6 GB consumer GPUs.</em></p>
</div>

<div align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/Lightning-2.2%2B-792ee5?logo=pytorch-lightning&logoColor=white" alt="PyTorch Lightning"></a>
  <a href="https://hydra.cc/"><img src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd" alt="Hydra"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Tracking-W%26B-FFBE00?logo=weightsandbiases&logoColor=black" alt="Weights & Biases"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
</div>

---

## Overview

**Hi-MambaSR** proposes a multi-paradigm generator that operates entirely in the latent space of a pre-trained variational autoencoder. The denoising backbone is a `UNet2DModel` whose deepest encoder stage, mid-block, and deepest decoder stage are augmented with:

| Injection Point | Module | Role |
|---|---|---|
| Deepest encoder block | **SwinBlock** (Flash-Attention, windowed shifted self-attention) | Captures precise local high-frequency textures |
| Mid-block | **HiMambaBottleneck** (bi-directional Mamba SSM + depth-wise conv spatial gate) | Provides linear-complexity global structural coherence |
| Deepest decoder block | **SwinBlock** | Reconstructs fine detail from skip connections |

A **Relativistic Patch-GAN discriminator** with Spectral Normalisation and Instance Normalisation drives adversarial training, while a multi-scale **VGG-19 perceptual loss** (with gradient checkpointing) and a **Sobel edge-aware loss** guide the generator toward natural image manifolds.

A **pixel-space residual skip** (bicubic up-sample + VAE residual) bypasses autoencoder blurring, and a 20-step **DDIM fast-sampling** trajectory replaces the full 1000-step DDPM chain at inference time.

<div align="center">
  <img src="figures/himambasr_architecture.png" width="85%" alt="Hi-MambaSR Architecture">
  <p><em>Figure 1 — End-to-end Hi-MambaSR pipeline: LR → Latent Encoding → Hybrid UNet Denoising → Latent Decoding + Pixel Residual → SR output.</em></p>
</div>

---

## Key Contributions

1. **6 GB VRAM-Hardened Training Pipeline**
   - Micro-batch size of 2 with gradient accumulation of 32 (effective batch = 64).
   - Gradient checkpointing in the VGG-19 feature extractor and the Mamba bottleneck to cap peak memory.
   - Sequential micro-batch VAE decoding during validation/test to avoid OOM.

2. **Stable Mamba Integration**
   - RMSNorm replaces LayerNorm to prevent mixed-precision instability in the SSM state matrix.
   - Linear warmup (10 % of total epochs) + global gradient clipping (norm ≤ 1.0) to suppress exploding gradients.

3. **Hybrid Attention–SSM Bottleneck**
   - Swin Transformer blocks at the deepest U-Net level for localised pixel-precise attention.
   - Bi-directional Mamba selective scan for *O(n)* global context aggregation — fused with a learnable depth-wise convolution spatial gate.

4. **Small-Batch GAN Stability**
   - Instance Normalisation in the discriminator prevents batch-statistics collapse at batch size 2–4.
   - Spectral Normalisation enforces Lipschitz continuity across all convolutional layers.

5. **Accelerated Inference**
   - Continuous 1000-step DDPM training schedule properly sub-sampled to a 20–50 step DDIM trajectory with correct `alpha_bar` interpolation.

<div align="center">
  <img src="figures/himamba_bottleneck_detail.png" width="70%" alt="HiMamba Bottleneck Detail">
  <p><em>Figure 2 — HiMambaBottleneck: RMSNorm → Bi-directional SSM ⊕ Sigmoid-gated DWConv → Learnable residual scale.</em></p>
</div>

---

## Project Structure

```
Hi-MambaSR/
├── Hi-MambaSR/                     # Core model package
│   ├── HiMambaSR.py                # PyTorch Lightning module (training, validation, test loops)
│   └── modules/
│       ├── UNet.py                 # HybridUNet, SwinBlock, MultiHeadSelectiveScan, HiMambaBottleneck
│       ├── Diffusion.py            # DDPM/DDIM noise schedules & sampling loops
│       ├── Discriminator.py        # Relativistic PatchGAN + ResNet50 discriminator (ablation)
│       ├── FeatureExtractor.py     # Multi-scale VGG-19 perceptual loss
│       └── VggLoss.py              # Single-scale Conv4_4 VGG loss (alternative)
├── scripts/
│   ├── data_loader.py              # CelebA-HQ / DIV2K data pipeline
│   ├── model_config.py             # Model & component construction from Hydra config
│   └── utilis.py                   # Path & naming utilities
├── conf/
│   └── config_mamba.yaml           # Default 6 GB-optimised Hydra configuration
├── train_model.py                  # Training entry point
├── evaluate_model.py               # Multi-step evaluation suite
├── generate_figures.py             # Publication figure generation
├── get_data.sh                     # Automated dataset download & LR generation
├── requirements-gpu.txt            # GPU training dependencies
├── requirements-data.txt           # Data-processing dependencies
├── CONFIGS.md                      # Configuration parameter reference
└── LICENSE
```

---

## Getting Started

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 (tested on 3.10) |
| PyTorch | ≥ 2.0 (`torch.compile`, Flash Attention) |
| CUDA | ≥ 11.8 (required by `mamba-ssm`) |
| GPU VRAM | ≥ 6 GB |

### Installation

```bash
# Clone the repository
git clone https://github.com/samarthya04/Hi-MambaSR.git
cd Hi-MambaSR

# Create a conda environment
conda create -n himamba python=3.10 -y
conda activate himamba

# Install data-processing dependencies
pip install -r requirements-data.txt

# Install GPU/training dependencies (installs PyTorch, Lightning, Mamba, etc.)
pip install -r requirements-gpu.txt

# Authenticate with Weights & Biases
wandb login
```

### Dataset Preparation

Hi-MambaSR uses **CelebA-HQ** by default. The provided script handles downloading, splitting, and 4× bicubic down-sampling:

```bash
bash get_data.sh -c
```

The script will:
1. Download the HQ dataset via the Kaggle API.
2. Organise images into `data/{train,val,test}/` splits.
3. Generate the corresponding 4× down-sampled LR images.

> **Note:** Ensure the Kaggle CLI is configured (`~/.kaggle/kaggle.json`) before running.

---

## Usage

### Training

All hyper-parameters are managed declaratively through [Hydra](https://hydra.cc/). The default configuration targets 6 GB VRAM GPUs:

```bash
python train_model.py -cn config_mamba
```

Key default settings (see [`conf/config_mamba.yaml`](conf/config_mamba.yaml)):

| Parameter | Value | Rationale |
|---|---|---|
| `dataset.batch_size` | 2 | Fits within 6 GB VRAM |
| `trainer.accumulate_grad_batches` | 32 | Effective batch size = 64 |
| `trainer.precision` | `bf16-mixed` | Reduces memory; stable with RMSNorm |
| `trainer.gradient_clip_val` | 1.0 | Prevents Mamba state-matrix divergence |
| `diffusion.beta_type` | `cosine` | Smoother noise schedule for latent diffusion |
| `checkpoint.monitor` | `val/LPIPS` | Optimises for perceptual fidelity |

#### Resuming / Fine-Tuning

```yaml
# conf/config_mamba.yaml
trainer:
  resume_from_checkpoint: "models/checkpoints/Hi-MambaSR-best.ckpt"
model:
  lr: 1e-5   # Lower LR for fine-tuning
```

```bash
python train_model.py -cn config_mamba
```

### Evaluation

```bash
python evaluate_model.py -cn config_mamba
```

Before running, set the model path and evaluation sweep in `conf/config_mamba.yaml`:

```yaml
model:
  load_model: "models/checkpoints/<your_best_checkpoint>.ckpt"
evaluation:
  mode: 'all'                          # 'all' | 'steps' | 'posterior'
  steps: [25, 50]                      # Timestep configurations to sweep
  posteriors: ['ddim']                 # Posterior types to sweep
  results_file: 'evaluation_results/hi_mambasr_benchmarks.csv'
```

The evaluation suite will:
- Run inference across every `(posterior, steps)` combination.
- Export metrics (PSNR, SSIM, LPIPS) to a CSV file.
- Log comparative bar-charts to Weights & Biases.

### Publication Figures

```bash
python generate_figures.py
```

Generates architecture diagrams, bottleneck schematics, and VRAM comparison charts to `figures/`.

---

## Architecture Details

### Generator — `HybridUNet`

```
Input: [x_t ∣ z_lr] ∈ ℝ^{B×8×H×W}         (noisy latent ∥ LR latent)
  │
  ├─ Encoder (UNet2DModel down_blocks)
  │     └── Deepest block wrapped with SwinBlock (Flash-Attention, window 8, shift 4)
  │
  ├─ Mid-Block → HiMambaBottleneck
  │     ├── RMSNorm
  │     ├── Bi-directional Mamba SSM (d_state=32, d_conv=4)
  │     ├── Sigmoid-gated DWConv spatial branch
  │     └── Learnable residual scale (initialised to 0)
  │
  └─ Decoder (UNet2DModel up_blocks)
        └── Deepest block wrapped with SwinBlock
        
Output: ε̂ ∈ ℝ^{B×4×H×W}                   (predicted noise)
```

### Discriminator — `Discriminator`

- **Type:** Relativistic PatchGAN (`BCEWithLogitsLoss`)
- **Input:** Concatenated `[SR ∣ LR]` or `[HR ∣ LR]` (6 channels)
- **Normalisation:** InstanceNorm2d (affine) + SpectralNorm on all convolutions
- **Channels:** `[64, 128, 256]` → AdaptiveAvgPool → `1×1` logit

### Loss Function

```
L_total = L_pixel  +  α_p · L_perceptual  +  α_a · L_adversarial  +  L_edge

L_pixel       = L1(SR, HR)
L_perceptual  = Σ_k w_k · L1(VGG_k(SR), VGG_k(HR))       (k ∈ {conv1_2 … conv5_4})
L_adversarial = BCE(D(SR|LR) − 𝔼[D(HR|LR)], 1)           (Relativistic average)
L_edge        = L1(Sobel(SR), Sobel(HR))
```

### Diffusion Engine

| Property | Training | Inference |
|---|---|---|
| Schedule | Cosine β, 1000 steps | Sub-sampled from training schedule |
| Posterior | DDPM (stochastic) | DDIM (deterministic, 20–50 steps) |
| Latent space | AutoencoderKL / AutoencoderTiny | Same encoder; micro-batch decoder |

---

## Metrics

The framework evaluates on the **Y channel** (YCbCr) following standard SR conventions:

| Metric | Description | Target |
|---|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio (dB) | ↑ Higher is better |
| **SSIM** | Structural Similarity Index | ↑ Higher is better |
| **LPIPS** | Learned Perceptual Image Patch Similarity | ↓ Lower is better |

Self-ensemble (8-fold geometric TTA) is supported at test time for an additional ~0.1–0.3 dB PSNR gain.

---

## Configuration Reference

See [`CONFIGS.md`](CONFIGS.md) for a complete parameter-by-parameter reference of all Hydra configuration options.

---

## Reproducibility

- Global random seed is fixed at `42` via `seed_everything(42, workers=True)`.
- `torch.set_float32_matmul_precision('medium')` is enabled for Ampere+ GPU acceleration.
- `deterministic=False` is used because `deterministic=True` is incompatible with Flash Attention and Mamba CUDA kernels.

---

## Acknowledgements

This work builds upon foundational research from:

- **Mamba: Linear-Time Sequence Modelling with Selective State Spaces** — Gu & Dao, 2023
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** — Liu et al., 2021
- **High-Resolution Image Synthesis with Latent Diffusion Models** — Rombach et al., 2022
- **Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data** — Wang et al., 2021
- **Improved Denoising Diffusion Probabilistic Models** — Nichol & Dhariwal, 2021

We also acknowledge the open-source ecosystems of [🤗 Diffusers](https://github.com/huggingface/diffusers), [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning), and [mamba-ssm](https://github.com/state-spaces/mamba).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you find Hi-MambaSR useful in your research, please consider citing:

```bibtex
@software{chattree2025himambasr,
  author    = {Chattree, Samarthya Earnest},
  title     = {{Hi-MambaSR}: Hierarchical State-Space Refinement for Latent Diffusion Super-Resolution},
  year      = {2026},
  url       = {https://github.com/samarthya04/Hi-MambaSR}
}
```
