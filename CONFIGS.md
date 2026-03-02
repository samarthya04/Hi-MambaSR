# Hi-MambaSR Configuration Documentation

This document provides a comprehensive explanation of the parameters used in the Hydra configuration files located in the `conf/` directory. 

The primary configuration file is `conf/config_mamba.yaml`, which is meticulously optimized to train the Hi-MambaSR architecture end-to-end within **6GB VRAM constraints** using 8-bit optimizers and gradient checkpointing.

---

## General Structure

Hi-MambaSR relies on [Hydra](https://hydra.cc/) for declarative experiment management. The configuration is divided into logical blocks:

- **Mode**: Defines the execution phase (`train`, `test`, `train-test`).
- **Model**: Hyper-parameters for loss weights, learning rates, and checkpoint loading.
- **Trainer**: PyTorch Lightning hardware targets, precision, memory optimizations, and accumulation control.
- **Dataset**: Image pipeline resolution and batching configurations.
- **Evaluation**: Inference steps and targets for automated benchmarking.
- **Architecture Blocks**: Specific layer counts and configurations for the Autoencoder, UNet Backbone, Diffusion Engine, and Discriminator.

---

## Detailed Parameter Reference

### **1. Mode & Global Settings**

| Parameter | Type | Description |
|-----------|---|-------------|
| `mode` | `str` | Execution mode. Options: <br/> `train`: Executes fit loop.<br/> `test`: Executes inference loop.<br/> `train-test`: Executes fit loop, serializes `.pth` on completion, and immediately runs benchmarking. |
| `use_perceptual_loss` | `bool` | Whether to compute multi-scale VGG feature loss. Default: `true`. |

---

### **2. Model Configuration (`model`)**

Dictates the architecture instantiation and gradient loss scaling.

| Parameter | Type | Description |
|-----------|---|-------------|
| `name` | `str` | Model variant. <br/> `Hi-MambaSR`: Primary Swin/Mamba hybrid framework.<br/> `SupResDiffGAN_without_adv`: Ablation model (no discriminator hook).<br/> `SupResDiffGAN_simple_gan`: Sub-variant for baseline comparison. |
| `lr` | `float` | Base learning rate for optimizer. Peak LR during linear warmup. Default: `2e-5`. |
| `alfa_perceptual` | `float` | Scale factor for VGG/LPIPS perceptual loss. Default: `0.08`. |
| `alfa_adv` | `float` | Scale factor for Relativistic PatchGAN adversarial loss. Default `1e-3`. |
| `load_model` | `str` | Path to restore weights. Can load both Lightning `.ckpt` dictionaries and clean `.pth` weight serializations. |

---

### **3. Trainer Environment (`trainer`)**

Hardware, throughput, and memory optimization configurations passed to `pytorch_lightning.Trainer`.

| Parameter | Type | Description |
|-----------|---|-------------|
| `max_epochs` | `int` | Maximum sweeps over the training dataset. Default: `100`. |
| `precision` | `str` | Calculation precision. `bf16-mixed` is required for VRAM savings without `fp16` gradient explosion/NaNs common to VAE latent gradients. |
| `accumulate_grad_batches` | `int` | Number of micro-batches to compute before an optimizer step. For `batch_size=2` and `accumulate_grad_batches=16`, the effective batch size is 32. Default: `16`. |
| `gradient_clip_val` | `float` | Maximum Global Gradient Norm. Set to `1.0` to prevent the Mamba Selective Scan from exploding on high frequency gradients. |
| `optimizer_8bit` | `bool` | Enables `bitsandbytes.optim.AdamW8bit`, quantizing optimizer states from fp32 to int8. Saves ~75% optimizer memory (~120MB on Hi-MambaSR). Default: `true`. |
| `benchmark` | `bool` | Set to `true` to allow cuDNN to find optimal convolution algorithms. *(Note: Determinism is explicitly disabled as it inherently conflicts with Flash Attention).* |

---

### **4. Dataset (`dataset`)**

Controls data loading hooks and PyTorch multiprocessing behaviors.

| Parameter | Type | Description |
|-----------|---|-------------|
| `name` | `str` | Target dataset identifier (e.g. `celeb`, `div2k`, `imagenet`). |
| `batch_size` | `int` | Physical micro-batch size. **Must be 2 or at most 4** to fit on a 6GB VRAM GPU. Default: `2`. |
| `resize` | `bool` | Whether to actively execute down-sampling generation in RAM via bicubic resampling interpolation. |
| `scale` | `int` | Super-Resolution upscale factor (e.g. `4`). |

---

### **5. Evaluation Benchmarks (`evaluation`)**

Parameters utilized by `evaluate_model.py` to auto-generate publication figures and CSV logs.

| Parameter | Type | Description |
|-----------|---|-------------|
| `mode` | `str` | Sweep execution range: <br/> `steps`: Sweep through `steps` array.<br/> `posterior`: Sweep `posteriors` array.<br/> `all`: Cross-product combinatorial sweep. |
| `steps` | `list` | Interpolated timestep configurations to test (e.g., `[25, 50]`). |
| `posteriors` | `list` | Reverse trajectories to compute (e.g., `['ddim']`). |
| `results_file`| `str` | Path/name to serialize the resultant `.csv` benchmark report. |

---

### **6. Architecture Dimensions**

Sets the structural capacity profiles of internal networks. Defaults are tuned heavily to optimize the VRAM floor.

#### **UNet Backbone (`unet`)**
- Example: `[64, 96, 128, 256]`
- Controls the block out-channels dimension list for the core generator. The Mamba/Swin bottleneck operates at the deepest latent projection (`256`).

#### **Diffusion Engine (`diffusion`)**
| Parameter | Type | Description |
|-----------|---|-------------|
| `timesteps` | `int` | Total base chain length. Default: `500`. Reduced from 1000 for faster training with minimal quality impact. |
| `beta_type` | `str` | Schedule algorithm. `cosine` prevents abrupt SNR destruction at the tail of the diffusion chain compared to `linear`. |
| `posterior_type` | `str` | Training posterior. `ddpm` (stochastic) for maximum diversity during training. |
| `validation_timesteps` | `int` | The evaluation subsampling target. Default: `25`. |
| `validation_posterior_type` | `str` | Default evaluation solver, usually `ddim` for determinism and speed. |

#### **Discriminator (`discriminator`)**
| Parameter | Type | Description |
|-----------|---|-------------|
| `in_channels` | `int` | Always `6`. Accounts for concatenating the predicted/ground_truth image `[3]` with the low-resolution condition `[3]`. |
| `channels` | `list` | Example: `[64, 128, 256]`. Hierarchical projection channels. Deeper arrays increase receptive field but rapidly consume memory. |

---

### **7. Logging & Checkpoints**

| Parameter | Application |
|---|---|
| `checkpoint.monitor` | Metric algorithm observes to save. Example: `val/LPIPS` |
| `checkpoint.mode` | `min` forces tracking for minimized LPIPS/VGG distances. |
| `checkpoint.save_top_k` | Number of best checkpoints to retain. Default: `3`. |
| `checkpoint.save_last` | Always save the latest checkpoint for resumption. Default: `true`. |
| `wandb.project` | Remote namespace for Weights & Biases telemetry suite. |

---

### **8. Memory Optimization Stack**

Hi-MambaSR uses a multi-layered approach to fit training within 6GB VRAM:

| Technique | Location | VRAM Savings |
|---|---|---|
| **8-bit Adam (bitsandbytes)** | `HiMambaSR.configure_optimizers()` | ~75% optimizer state memory (~120MB) |
| **Gradient Checkpointing** | SwinBlock attention, Mamba SSM, UNet backbone | ~50-70% activation memory |
| **VAE Tiling + Slicing** | `AutoencoderKL.enable_tiling/slicing()` | Reduces peak VAE decode memory |
| **Micro-batch VAE Decode** | `micro_batch_decode()` in training step | Sequential decoding caps VRAM spikes |
| **bf16-mixed Precision** | PyTorch Lightning precision plugin | ~50% reduction in activation/weight memory |
| **Frozen Components** | VAE encoder/decoder, LPIPS backbone | Eliminates gradient storage for ~86M params |
| **Cached Discriminator Inputs** | Reuses decoded images from generator block | Eliminates 2 redundant VAE decodes/step |
