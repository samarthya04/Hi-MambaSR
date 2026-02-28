import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import wandb
import time
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class HiMambaSR(pl.LightningModule):
    """
    Hi-MambaSR: Hierarchical State-Space Refinement for Latent Diffusion Super-Resolution.
    Hardened for 6GB VRAM with Latent Regularization and Micro-Batch Decoding.
    """

    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 2e-2,
        alfa_adv: float = 5e-3,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super(HiMambaSR, self).__init__()
        self.save_hyperparameters(ignore=["ae", "discriminator", "unet", "diffusion", "vgg_loss"])
        
        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion
        self.vgg_loss = vgg_loss

        # L1 produces sharper gradients than MSE for pixel/latent reconstruction
        self.content_loss = nn.L1Loss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        # Freeze LPIPS backbone — it's a metric, not a trainable loss component.
        # Without this, AlexNet accumulates gradients, wasting VRAM and risking metric drift.
        for param in self.lpips.parameters():
            param.requires_grad = False

        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        # Freeze the ENTIRE VAE (encoder + decoder)
        # Unfreezing the decoder is destabilizing: the skip connection already handles
        # high-frequency pixel-space detail, and the decoder's pre-trained weights produce
        # cleaner outputs than partially fine-tuned ones on a small dataset.
        for param in self.ae.parameters():
            param.requires_grad = False

        # Store reference to raw (uncompiled) generator for EMA + checkpointing.
        # torch.compile wraps the module in an opaque _dynamo object — deepcopy-ing
        # the compiled wrapper is unreliable and cannot always be serialized.
        self._raw_generator = unet

        # Compile U-Net/Mamba Generator for faster training throughput.
        if hasattr(torch, 'compile'):
            self.generator = torch.compile(self.generator)

        self.automatic_optimization = False
        self.accumulate_grad_batches = 32  # Manual grad accumulation (batch_size=2 × 32 = 64 effective)
        self.test_step_outputs = []
        
        # EMA (Exponential Moving Average) for generator weights
        # This creates a smoothed copy of the generator that is used at inference time.
        # EMA weights are less noisy than per-step weights, producing more stable
        # and higher-quality outputs. Standard in ESRGAN, Real-ESRGAN, SwinIR.
        self.ema_decay = 0.999
        self.ema_generator = None  # Initialized lazily on first training step
        
        # Noise difficulty progression (EMA)
        self.ema_weight = 0.99
        self.ema_mean = 0.5
        self.s = 0

        # VAE VRAM Optimizations
        self.ae.enable_slicing()
        self.ae.enable_tiling()
        
        self._register_sobel_kernels()

    def _register_sobel_kernels(self):
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_kx', kx)
        self.register_buffer('sobel_ky', ky)

    def normalize_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale from [-1, 1] to [0, 1] for LPIPS (initialized with normalize=True)."""
        return torch.clamp((x.float() + 1.0) / 2.0, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference loop: Encode -> Refine -> Decode (uses EMA weights if available)."""
        # Use EMA generator for inference if available (stabilizes output quality)
        gen = self.ema_generator if (self.ema_generator is not None and not self.training) else self.generator
        
        with torch.no_grad():
            posterior = self.ae.encode(x).latent_dist
            x_lat = posterior.mode() * self.ae.config.scaling_factor
        
        # Use DDIM with fewer timesteps for faster, higher quality inference
        orig_timesteps = self.diffusion.timesteps
        orig_posterior = self.diffusion.posterior_type
        self.diffusion.set_timesteps(25)
        self.diffusion.set_posterior_type('ddim')
        
        x_gen = self.diffusion.sample(gen, x_lat, x_lat.shape)
        
        # Restore original settings
        self.diffusion.set_timesteps(orig_timesteps)
        self.diffusion.set_posterior_type(orig_posterior)
        
        with torch.no_grad():
            x_out = self.ae.decode(x_gen.to(torch.float32) / self.ae.config.scaling_factor).sample.to(x_lat.dtype)
            
        # High-Frequency Skip Connection (Pixel-Space Residual)
        x_upsampled = F.interpolate(x, size=x_out.shape[-2:], mode='bicubic', align_corners=False)
        return torch.clamp(x_out + x_upsampled, -1, 1)

    def micro_batch_decode(self, latent_tensor: torch.Tensor, micro_batch_size: int = 1) -> torch.Tensor:
        """Sequential decoding to limit peak VRAM on 6GB GPUs."""
        scale = self.ae.config.scaling_factor
        decoded_chunks = []
        for i in range(0, latent_tensor.shape[0], micro_batch_size):
            chunk = latent_tensor[i : i + micro_batch_size]
            # Force fp32 during decoding to prevent NaNs/black images in mixed precision
            decoded = self.ae.decode(chunk.to(torch.float32) / scale).sample.to(latent_tensor.dtype)
            decoded_chunks.append(decoded)
        return torch.cat(decoded_chunks, dim=0)

    def calculate_edge_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        def get_gradients(x):
            x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
            c = x.size(1)
            gx = F.conv2d(x_padded, self.sobel_kx.repeat(c, 1, 1, 1), groups=c)
            gy = F.conv2d(x_padded, self.sobel_ky.repeat(c, 1, 1, 1), groups=c)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)
        # Detach HR gradients — edge loss should only guide the SR output
        with torch.no_grad():
            hr_grads = get_gradients(hr)
        return F.l1_loss(get_gradients(sr), hr_grads)

    def training_step(self, batch: dict, batch_idx: int):
        lr_img, hr_img = batch["lr"], batch["hr"]
        
        # Initialize EMA from raw (uncompiled) weights on first step
        if self.ema_generator is None:
            self.ema_generator = copy.deepcopy(self._raw_generator)
            self.ema_generator.eval()
            for p in self.ema_generator.parameters():
                p.requires_grad = False
        
        # --- Data Augmentation (Random Flips) ---
        # Applied identically to both LR and HR to maintain spatial alignment
        if torch.rand(1).item() > 0.5:
            lr_img = torch.flip(lr_img, dims=[-1])  # Horizontal flip
            hr_img = torch.flip(hr_img, dims=[-1])
        if torch.rand(1).item() > 0.5:
            lr_img = torch.flip(lr_img, dims=[-2])  # Vertical flip
            hr_img = torch.flip(hr_img, dims=[-2])
        
        optimizer_g, optimizer_d = self.optimizers()
        scale = self.ae.config.scaling_factor
        
        # Manual gradient accumulation flag
        is_accumulation_step = (batch_idx + 1) % self.accumulate_grad_batches == 0

        # 1. Feature Extraction (Latent Space)
        # Using .mode() (deterministic mean) instead of .sample() (stochastic)
        # .sample() injects random noise from the VAE posterior on every forward pass,
        # meaning the same image produces different latent targets each epoch.
        # This forces the Generator to chase a moving target, severely slowing convergence.
        with torch.no_grad():
            lr_lat = self.ae.encode(lr_img).latent_dist.mode().detach() * scale
            x0_lat = self.ae.encode(hr_img).latent_dist.mode().detach() * scale

        # 2. Diffusion logic
        t = torch.randint(0, self.diffusion.timesteps, (x0_lat.shape[0],), device=self.device)
        x_t = self.diffusion.forward(x0_lat, t)
        alfa_bars = self.diffusion.alpha_bars_torch.to(self.device)[t]
        
        # 3. Generator Core Pass
        x_gen_0 = self.generator(lr_lat, x_t, alfa_bars)

        # --- GENERATOR UPDATE ---
        # Gradients are accumulated over `accumulate_grad_batches` micro-batches.
        # optimizer.step() and zero_grad() only fire at accumulation boundaries.
        self.toggle_optimizer(optimizer_g)
        
        # Zero gradients only at the start of each accumulation window
        if batch_idx % self.accumulate_grad_batches == 0:
            optimizer_g.zero_grad(set_to_none=True)
        
        l_content = self.content_loss(x_gen_0, x0_lat)
        
        # --- LATENT REGULARIZATION (Soft L1) ---
        # L1 is softer than L2 for regularization; L2 penalizes large values quadratically
        # which over-suppresses high-magnitude latent activations that encode fine detail.
        l_lat_reg = torch.mean(torch.abs(x_gen_0))
        
        # Micro-Batch Decode for Perceptual Loss (with Skip Connection)
        sr_decoded = self.micro_batch_decode(x_gen_0)
        lr_upsampled = F.interpolate(lr_img, size=sr_decoded.shape[-2:], mode='bicubic', align_corners=False)
        sr_img = torch.clamp(sr_decoded + lr_upsampled, -1, 1)
        
        l_lpips = self.lpips(self.normalize_for_lpips(sr_img), self.normalize_for_lpips(hr_img))
        l_edge = self.calculate_edge_loss(sr_img, hr_img)
        
        # VGG/FeatureExtractor perceptual loss (multi-scale)
        l_vgg = self.vgg_loss(sr_img, hr_img) if self.vgg_loss else 0.0

        # --- Relativistic GAN loss (detached from generator graph for 6GB VRAM) ---
        # On 6GB VRAM, maintaining 3 VAE decoder computation graphs simultaneously
        # causes OOM. The adversarial signal (α_adv=1e-3) is marginal compared to
        # LPIPS+VGG+edge losses. Detaching here keeps VRAM bounded while the
        # discriminator still trains properly in its own update block below.
        with torch.no_grad():
            s_tensor = torch.full((x0_lat.shape[0],), self.s, device=self.device, dtype=torch.long)
            
            sr_s_decoded = self.micro_batch_decode(self.diffusion.forward(x_gen_0, s_tensor))
            sr_s_img = torch.clamp(sr_s_decoded + lr_upsampled, -1, 1)
            
            hr_s_img = torch.clamp(self.micro_batch_decode(self.diffusion.forward(x0_lat, s_tensor)), -1, 1)
            
            y = torch.randint(0, 2, (hr_s_img.shape[0],), device=self.device).view(-1, 1, 1, 1)
            first = torch.where(y == 0, hr_s_img, sr_s_img)
            second = torch.where(y == 0, sr_s_img, hr_s_img)
            
            prediction = self.discriminator(torch.cat([first, second], dim=1))
            l_adv = self.adversarial_loss(prediction, 1.0 - y.float())

        # Composite Loss
        # l_content: latent-space L1 reconstruction (gradient ✓)
        # l_lpips: direct perceptual similarity (gradient ✓)
        # l_vgg: multi-scale VGG feature matching (gradient ✓)
        # l_adv: adversarial realism (logged only, no gradient to generator)
        # l_edge: Sobel edge preservation (gradient ✓)
        # l_lat_reg: soft latent magnitude regularization (gradient ✓)
        g_loss = l_content \
                 + (self.hparams.alfa_perceptual * (l_lpips + l_vgg)) \
                 + (0.05 * l_edge) \
                 + (5e-4 * l_lat_reg)
        
        # Scale loss by accumulation count for correct gradient magnitude
        self.manual_backward(g_loss / self.accumulate_grad_batches)
        
        # Step optimizer only at accumulation boundary
        if is_accumulation_step:
            self.clip_gradients(optimizer_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        # Update EMA weights on the raw (uncompiled) parameter tensors
        if self.ema_generator is not None:
            with torch.no_grad():
                for ema_p, raw_p in zip(
                    self.ema_generator.parameters(),
                    self._raw_generator.parameters()
                ):
                    ema_p.data.mul_(self.ema_decay).add_(raw_p.data, alpha=1.0 - self.ema_decay)
        
        self.log_dict({
            "train/g_loss": g_loss, 
            "train/l_content": l_content,
            "train/g_lpips": l_lpips,
            "train/l_edge": l_edge,
            "train/l_vgg": l_vgg if isinstance(l_vgg, torch.Tensor) else 0.0,
            "train/l_adv": l_adv,
            "train/l_lat_reg": l_lat_reg,
        }, prog_bar=True, batch_size=lr_img.shape[0])
        self.calculate_ema_noise_step(prediction.detach(), y.float())

        # --- DISCRIMINATOR UPDATE (Every Other Step, with Accumulation) ---
        if batch_idx % 2 == 0:
            self.toggle_optimizer(optimizer_d)
            
            # Zero gradients only at the start of each accumulation window
            if batch_idx % self.accumulate_grad_batches == 0:
                optimizer_d.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                s_tensor = torch.full((x0_lat.shape[0],), self.s, device=self.device, dtype=torch.long)
                
                sr_s_decoded = self.micro_batch_decode(self.diffusion.forward(x_gen_0.detach(), s_tensor))
                lr_upsampled_d = F.interpolate(lr_img, size=sr_s_decoded.shape[-2:], mode='bicubic', align_corners=False)
                sr_s_img = torch.clamp(sr_s_decoded + lr_upsampled_d, -1, 1)
                
                hr_s_img = torch.clamp(self.micro_batch_decode(self.diffusion.forward(x0_lat, s_tensor)), -1, 1)
            
            y = torch.randint(0, 2, (hr_s_img.shape[0],), device=self.device).view(-1, 1, 1, 1)
            first = torch.where(y == 0, hr_s_img, sr_s_img)
            second = torch.where(y == 0, sr_s_img, hr_s_img)
            
            prediction = self.discriminator(torch.cat([first, second], dim=1))
            d_loss = self.adversarial_loss(prediction, y.float())
            
            self.manual_backward(d_loss / self.accumulate_grad_batches)
            
            if is_accumulation_step:
                self.clip_gradients(optimizer_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)
            self.log("train/d_loss", d_loss, prog_bar=True, batch_size=lr_img.shape[0])

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    def calculate_ema_noise_step(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        acc = ((torch.sigmoid(pred) > 0.5).float() == y).float().mean().item()
        self.ema_mean = acc * (1 - self.ema_weight) + self.ema_mean * self.ema_weight
        self.s = int(np.clip((self.ema_mean - 0.5) * 2 * self.diffusion.timesteps, 0, self.diffusion.timesteps - 1))
        self.log_dict({"train/ema_s": self.s, "train/ema_acc": acc}, on_epoch=True, batch_size=pred.shape[0])

    def configure_optimizers(self):
        gen_params = list(self.generator.parameters())
            
        # AdamW decouples weight decay from the gradient update, preventing
        # the LR warmup + cosine decay from interfering with regularization.
        opt_g = torch.optim.AdamW(gen_params, lr=self.lr, betas=self.betas, weight_decay=1e-4)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=self.lr * 0.5, betas=self.betas, weight_decay=1e-4)
        
        t_max = self.trainer.max_epochs 
        warmup_epochs = max(1, t_max // 10)
        
        warmup_g = torch.optim.lr_scheduler.LinearLR(opt_g, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=t_max - warmup_epochs, eta_min=1e-7)
        sch_g = torch.optim.lr_scheduler.SequentialLR(opt_g, schedulers=[warmup_g, cosine_g], milestones=[warmup_epochs])

        warmup_d = torch.optim.lr_scheduler.LinearLR(opt_d, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=t_max - warmup_epochs, eta_min=1e-7)
        sch_d = torch.optim.lr_scheduler.SequentialLR(opt_d, schedulers=[warmup_d, cosine_d], milestones=[warmup_epochs])

        return [opt_g, opt_d], [sch_g, sch_d]

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        self.log("lr/gen", sch_g.get_last_lr()[0], on_epoch=True, prog_bar=True)
        sch_g.step()
        sch_d.step()

    @staticmethod
    def _rgb_to_ycbcr_y(img_np: np.ndarray) -> np.ndarray:
        """Convert RGB [0,1] numpy array (H,W,3) to Y channel. Research standard for PSNR/SSIM."""
        return 16./255. + (65.481/255. * img_np[..., 0] + 128.553/255. * img_np[..., 1] + 24.966/255. * img_np[..., 2])

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        lr_img, hr_img = batch["lr"], batch["hr"]
        sr_img = self(lr_img)
        padding_info = {"lr": batch["padding_data_lr"], "hr": batch["padding_data_hr"]}

        # Convert to [0,1] numpy for metric computation
        hr_np = np.clip((hr_img.float().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2, 0, 1)
        sr_np = np.clip((sr_img.float().cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2, 0, 1)

        # Y-channel metrics (research standard: crop border by scale factor)
        border = self.hparams.get('scale', 4) if hasattr(self.hparams, 'get') else 4
        psnr_vals, ssim_vals = [], []
        for i in range(hr_np.shape[0]):
            hr_y = self._rgb_to_ycbcr_y(hr_np[i])
            sr_y = self._rgb_to_ycbcr_y(sr_np[i])
            # Crop border pixels (standard in SR evaluation)
            if border > 0:
                hr_y = hr_y[border:-border, border:-border]
                sr_y = sr_y[border:-border, border:-border]
            psnr_vals.append(peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0))
            ssim_vals.append(structural_similarity(hr_y, sr_y, data_range=1.0))
        
        val_psnr = float(np.mean(psnr_vals))
        val_ssim = float(np.mean(ssim_vals))
        val_lpips = self.lpips(self.normalize_for_lpips(hr_img), self.normalize_for_lpips(sr_img)).cpu().item()
        
        if batch_idx == 0:
            per_metrics = [(val_psnr, val_ssim, val_lpips)]
            img_grid = self.plot_images_with_metrics(hr_img.float(), lr_img.float(), sr_img.float(), padding_info, 
                                                   f"Hi-MambaSR | Epoch {self.current_epoch}", per_metrics)
            self.logger.experiment.log({"val_samples": wandb.Image(img_grid)})

        self.log_dict({
            "val/PSNR": val_psnr, 
            "val/SSIM": val_ssim, 
            "val/LPIPS": val_lpips
        }, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=hr_img.shape[0])

    def plot_images_with_metrics(self, hr_img, lr_img, sr_img, padding_info, title, per_image_metrics) -> np.ndarray:
        num_samples = min(2, hr_img.shape[0])
        # Layout: num_samples columns for images + 1 column for metrics panel
        fig, axs = plt.subplots(3, num_samples + 1, figsize=(4 * (num_samples + 1), 12), dpi=120,
                                gridspec_kw={'width_ratios': [1] * num_samples + [0.8]})
        fig.patch.set_facecolor("#121212")
        fig.suptitle(title, fontsize=16, color="white", fontweight="bold", y=0.98)

        # --- Image columns ---
        row_labels = ["Ground Truth", "Bicubic Input", "Hi-MambaSR Prediction"]
        for i in range(num_samples):
            img_sets = [(hr_img[i], row_labels[0], axs[0, i]),
                        (lr_img[i], row_labels[1], axs[1, i]),
                        (sr_img[i], row_labels[2], axs[2, i])]
            for img_t, lbl, ax in img_sets:
                img_np = np.clip((img_t.detach().cpu().float().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)
                ax.imshow(img_np)
                ax.set_title(lbl, fontsize=11, color="white", pad=6)
                ax.axis('off')

        # --- Metrics panel (right column) ---
        metric_names = ["PSNR (dB)", "SSIM", "LPIPS"]
        # Color coding: higher is better for PSNR/SSIM, lower is better for LPIPS
        metric_colors = ["#4FC3F7", "#81C784", "#FF8A65"]

        for row_idx in range(3):
            ax = axs[row_idx, num_samples]
            ax.set_facecolor("#1E1E1E")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            if row_idx == 0:
                # Title for metrics panel
                ax.text(0.5, 0.92, "Validation Metrics", fontsize=12, color="#EEEEEE",
                        ha='center', va='top', fontweight='bold',
                        transform=ax.transAxes)
                # Display per-sample metrics
                for s_idx in range(min(num_samples, len(per_image_metrics))):
                    psnr_v, ssim_v, lpips_v = per_image_metrics[s_idx]
                    y_start = 0.72 - s_idx * 0.38
                    ax.text(0.5, y_start, f"Sample {s_idx + 1}", fontsize=10, color="#BDBDBD",
                            ha='center', va='top', fontstyle='italic', transform=ax.transAxes)
                    for m_i, (m_name, m_val, m_col) in enumerate(zip(
                            metric_names, [psnr_v, ssim_v, lpips_v], metric_colors)):
                        y_pos = y_start - 0.08 * (m_i + 1)
                        ax.text(0.5, y_pos, f"{m_name}: {m_val:.4f}", fontsize=11, color=m_col,
                                ha='center', va='top', fontweight='bold',
                                fontfamily='monospace', transform=ax.transAxes)
            elif row_idx == 1:
                # Arrow legend explaining metric directions
                ax.text(0.5, 0.7, "↑ PSNR  ↑ SSIM", fontsize=10, color="#81C784",
                        ha='center', va='center', transform=ax.transAxes)
                ax.text(0.5, 0.45, "↓ LPIPS (lower = better)", fontsize=10, color="#FF8A65",
                        ha='center', va='center', transform=ax.transAxes)
            else:
                # Epoch badge in bottom-right cell
                ax.text(0.5, 0.5, f"Epoch {self.current_epoch}",
                        fontsize=14, color="#FFF176", ha='center', va='center',
                        fontweight='bold', transform=ax.transAxes,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#333333',
                                  edgecolor='#FFF176', alpha=0.9))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    def _self_ensemble(self, lr_img: torch.Tensor) -> torch.Tensor:
        """Self-ensemble (TTA): average predictions over 8 geometric transforms.
        Standard technique in SR papers to boost PSNR/SSIM by ~0.1-0.3 dB."""
        outputs = []
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                for transpose in [False, True]:
                    x = lr_img
                    if flip_h:
                        x = torch.flip(x, [-1])
                    if flip_v:
                        x = torch.flip(x, [-2])
                    if transpose:
                        x = x.permute(0, 1, 3, 2)
                    
                    pred = self(x)
                    
                    # Reverse transforms
                    if transpose:
                        pred = pred.permute(0, 1, 3, 2)
                    if flip_v:
                        pred = torch.flip(pred, [-2])
                    if flip_h:
                        pred = torch.flip(pred, [-1])
                    outputs.append(pred)
        return torch.stack(outputs).mean(0)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        lr_img, hr_img = batch["lr"].float(), batch["hr"].float()
        start = time.perf_counter()
        # Self-ensemble for best test metrics (paper-quality numbers)
        sr_img = self._self_ensemble(lr_img).float()
        inf_time = time.perf_counter() - start
        
        # Y-channel metrics (matches validation)
        hr_np = np.clip((hr_img.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2, 0, 1)
        sr_np = np.clip((sr_img.cpu().numpy().transpose(0, 2, 3, 1) + 1) / 2, 0, 1)
        
        border = 4
        psnr_vals, ssim_vals = [], []
        for i in range(hr_np.shape[0]):
            hr_y = self._rgb_to_ycbcr_y(hr_np[i])
            sr_y = self._rgb_to_ycbcr_y(sr_np[i])
            if border > 0:
                hr_y = hr_y[border:-border, border:-border]
                sr_y = sr_y[border:-border, border:-border]
            psnr_vals.append(peak_signal_noise_ratio(hr_y, sr_y, data_range=1.0))
            ssim_vals.append(structural_similarity(hr_y, sr_y, data_range=1.0))
        
        psnr = float(np.mean(psnr_vals))
        ssim = float(np.mean(ssim_vals))
        lpips = self.lpips(self.normalize_for_lpips(hr_img), self.normalize_for_lpips(sr_img)).cpu().item()
        
        result = {"test/PSNR": psnr, "test/SSIM": ssim, "test/LPIPS": lpips, "test/inference_time": inf_time}
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs: return
        avg_res = {k: np.mean([x[k] for x in self.test_step_outputs]) for k in self.test_step_outputs[0].keys()}
        self.log_dict(avg_res, on_epoch=True, prog_bar=True)
        self.test_step_outputs.clear()
        
        # Log model complexity (for paper's efficiency table)
        total_params = sum(p.numel() for p in self.generator.parameters())
        trainable_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Hi-MambaSR Model Complexity Report")
        print(f"{'='*60}")
        print(f"Generator Parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
        print(f"Discriminator Parameters: {sum(p.numel() for p in self.discriminator.parameters()) / 1e6:.2f}M")
        print(f"{'='*60}\n")