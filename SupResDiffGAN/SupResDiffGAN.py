import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import time


class SupResDiffGAN(pl.LightningModule):
    """
    Optimized SupResDiffGAN for Super-Resolution.
    Integrates LPIPS perceptual loss, Edge-Aware loss, and dynamic EMA noise scheduling.
    """

    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 1e-2, 
        alfa_adv: float = 5e-3,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super(SupResDiffGAN, self).__init__()
        self.save_hyperparameters(ignore=["ae", "discriminator", "unet", "diffusion", "vgg_loss"])
        
        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion
        self.vgg_loss = vgg_loss

        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Using AlexNet for faster and more memory-efficient perceptual loss
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)

        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        for param in self.ae.parameters():
            param.requires_grad = False

        self.automatic_optimization = False
        self.test_step_outputs = []
        
        self.ema_weight = 0.98
        self.ema_mean = 0.5
        self.s = 0

    def normalize_for_lpips(self, x: torch.Tensor) -> torch.Tensor:
        """Rescales a tensor from [-1, 1] to [0, 1] for LPIPS compatibility."""
        return torch.clamp((x.float() + 1) / 2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_lat = (self.ae.encode(x).latents.detach() * self.ae.scaling_factor)
        x_gen = self.diffusion.sample(self.generator, x_lat, x_lat.shape)
        with torch.no_grad():
            x_out = self.ae.decode(x_gen / self.ae.scaling_factor).sample
        return torch.clamp(x_out, -1, 1)

    def get_edge_loss(self, sr, hr):
        """Calculates Sobel edge loss to prioritize high-frequency reconstruction."""
        def sobel_op(x):
            # Simple gradient filters
            dx = x[:, :, :, :-1] - x[:, :, :, 1:]
            dy = x[:, :, :-1, :] - x[:, :, 1:, :]
            return torch.abs(dx).mean() + torch.abs(dy).mean()
        return self.content_loss(sobel_op(sr), sobel_op(hr))

    def training_step(self, batch: dict, batch_idx: int):
        lr_img, hr_img = batch["lr"], batch["hr"]
        optimizer_g, optimizer_d = self.optimizers()

        with torch.no_grad():
            lr_lat = self.ae.encode(lr_img).latents.detach() * self.ae.scaling_factor
            x0_lat = self.ae.encode(hr_img).latents.detach() * self.ae.scaling_factor

        timesteps = torch.randint(0, self.diffusion.timesteps, (x0_lat.shape[0],), device=self.device)
        x_t = self.diffusion.forward(x0_lat, timesteps)
        alfa_bars = self.diffusion.alpha_bars_torch.to(self.device)[timesteps]
        x_gen_0 = self.generator(lr_lat, x_t, alfa_bars)

        s_tensor = torch.full((x0_lat.shape[0],), self.s, device=self.device, dtype=torch.long)
        x_s = self.diffusion.forward(x0_lat, s_tensor)
        x_gen_s = self.diffusion.forward(x_gen_0, s_tensor)

        with torch.no_grad():
            sr_img = torch.clamp(self.ae.decode(x_gen_0 / self.ae.scaling_factor).sample, -1, 1)
            hr_s_img = torch.clamp(self.ae.decode(x_s / self.ae.scaling_factor).sample, -1, 1)
            sr_s_img = torch.clamp(self.ae.decode(x_gen_s / self.ae.scaling_factor).sample, -1, 1)

        if batch_idx % 2 == 0:
            optimizer_g.zero_grad()
            g_loss = self.generator_loss(x0_lat, x_gen_0, hr_img, sr_img, hr_s_img, sr_s_img)
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.log("train/g_loss", g_loss, prog_bar=True, sync_dist=True)
        else:
            optimizer_d.zero_grad()
            d_loss = self.discriminator_loss(hr_s_img, sr_s_img.detach())
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.log("train/d_loss", d_loss, prog_bar=True, sync_dist=True)

    def generator_loss(self, x0, x_gen, hr_img, sr_img, hr_s_img, sr_s_img) -> torch.Tensor:
        content_loss = self.content_loss(x_gen, x0)
        
        # Perceptual Loss with float casting
        perceptual_loss = self.lpips(self.normalize_for_lpips(sr_img), self.normalize_for_lpips(hr_img))
        if self.vgg_loss is not None:
            perceptual_loss += self.vgg_loss(sr_img, hr_img)
            
        # Frequency edge constraint
        edge_loss = self.get_edge_loss(sr_img, hr_img)

        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=self.device).view(-1, 1, 1, 1)
        first = torch.where(y == 0, sr_s_img, hr_s_img)
        second = torch.where(y == 0, hr_s_img, sr_s_img)
        prediction = self.discriminator(torch.cat([first, second], dim=1))
        adversarial_loss = self.adversarial_loss(prediction, 1.0 - y.float())

        total_loss = (content_loss + self.alfa_perceptual * perceptual_loss + 
                      self.alfa_adv * adversarial_loss + 0.05 * edge_loss)
        
        self.log_dict({"train/g_content": content_loss, "train/g_lpips": perceptual_loss, "train/g_edge": edge_loss}, on_epoch=True)
        self.calculate_ema_noise_step(prediction, 1.0 - y.float())
        return total_loss

    def discriminator_loss(self, hr_s_img: torch.Tensor, sr_s_img: torch.Tensor) -> torch.Tensor:
        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=self.device).view(-1, 1, 1, 1)
        first = torch.where(y == 0, hr_s_img, sr_s_img)
        second = torch.where(y == 0, sr_s_img, hr_s_img)
        prediction = self.discriminator(torch.cat([first, second], dim=1))
        d_loss = self.adversarial_loss(prediction, y.float())
        self.calculate_ema_noise_step(prediction, y.float())
        return d_loss

    def calculate_ema_noise_step(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        acc = ((torch.sigmoid(pred) > 0.5).float() == y).float().mean().item()
        self.ema_mean = acc * (1 - self.ema_weight) + self.ema_mean * self.ema_weight
        self.s = int(np.clip((self.ema_mean - 0.5) * 2 * self.diffusion.timesteps, 0, self.diffusion.timesteps - 1))
        self.log_dict({"train/ema_s": self.s, "train/ema_acc": acc}, on_epoch=True)

    def configure_optimizers(self):
        # Using a slight weight decay for GAN stability
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-6)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas, weight_decay=1e-6)
        return [opt_g, opt_d]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        sr_img = self(lr_img)
        padding_info = {"lr": batch["padding_data_lr"], "hr": batch["padding_data_hr"]}

        hr_f32 = hr_img.float()
        sr_f32 = sr_img.float()

        if batch_idx < 1:
            try:
                per_image_metrics = []
                for i in range(min(3, lr_img.shape[0])):
                    hr_np = (hr_f32[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
                    sr_np = (sr_f32[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2
                    hr_np = np.clip(hr_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :], 0, 1).astype(np.float32)
                    sr_np = np.clip(sr_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :], 0, 1).astype(np.float32)
                    
                    psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
                    ssim = structural_similarity(hr_np, sr_np, channel_axis=-1, data_range=1.0)
                    lpips_val = self.lpips(self.normalize_for_lpips(hr_f32[i: i + 1]), 
                                          self.normalize_for_lpips(sr_f32[i: i + 1])).cpu().item()
                    per_image_metrics.append((psnr, ssim, lpips_val))

                img_array = self.plot_images_with_metrics(hr_f32, lr_img.float(), sr_f32, padding_info, 
                                                       f"Val Epoch {self.current_epoch}", per_image_metrics)
                self.logger.experiment.log({"validation_images": wandb.Image(img_array)})
            except Exception as e:
                print(f"Visualization error: {str(e)}")

        metrics = {"PSNR": [], "SSIM": []}
        for i in range(lr_img.shape[0]):
            hr_np = np.clip((hr_f32[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1).astype(np.float32)
            sr_np = np.clip((sr_f32[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1).astype(np.float32)
            hr_np = hr_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
            sr_np = sr_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
            metrics["PSNR"].append(peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0))
            metrics["SSIM"].append(structural_similarity(hr_np, sr_np, channel_axis=-1, data_range=1.0))

        lpips = self.lpips(self.normalize_for_lpips(hr_f32), self.normalize_for_lpips(sr_f32)).cpu().item()
        self.log_dict({"val/PSNR": np.mean(metrics["PSNR"]), "val/SSIM": np.mean(metrics["SSIM"]), 
                       "val/LPIPS": lpips}, on_epoch=True, prog_bar=True, sync_dist=True)

    def plot_images_with_metrics(self, hr_img, lr_img, sr_img, padding_info, title, per_image_metrics) -> np.ndarray:
        fig, axs = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
        fig.patch.set_facecolor("#1e1e1e")
        for i in range(min(3, len(per_image_metrics))):
            psnr, ssim, lpips_val = per_image_metrics[i]
            img_sets = [(hr_img[i], "HR GT", axs[0, i]), (lr_img[i], "LR Input", axs[1, i]), 
                        (sr_img[i], f"SR Pred\nLPIPS: {lpips_val:.3f}", axs[2, i])]
            for img_t, lbl, ax in img_sets:
                img_np = np.clip((img_t.detach().cpu().float().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1).astype(np.float32)
                ax.imshow(img_np)
                ax.set_title(lbl, fontsize=9, color="white")
                ax.axis('off')
        plt.tight_layout()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        lr_img, hr_img = batch["lr"].to(self.device).float(), batch["hr"].to(self.device).float()
        padding_info = {"hr": batch["padding_data_hr"]}
        start_time = time.perf_counter()
        sr_img = self(lr_img).float()
        elapsed_time = time.perf_counter() - start_time
        metrics = {"PSNR": [], "SSIM": []}
        for i in range(lr_img.shape[0]):
            hr_np = np.clip((hr_img[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1).astype(np.float32)
            sr_np = np.clip((sr_img[i].detach().cpu().numpy().transpose(1, 2, 0) + 1) / 2, 0, 1).astype(np.float32)
            hr_np = hr_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
            sr_np = sr_np[: padding_info["hr"][i][1], : padding_info["hr"][i][0], :]
            metrics["PSNR"].append(peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0))
            metrics["SSIM"].append(structural_similarity(hr_np, sr_np, channel_axis=-1, data_range=1.0))
        lpips = self.lpips(self.normalize_for_lpips(hr_img), self.normalize_for_lpips(sr_img)).cpu().item()
        result = {"PSNR": np.mean(metrics["PSNR"]), "SSIM": np.mean(metrics["SSIM"]), "LPIPS": lpips, "time": elapsed_time}
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self) -> None:
        avg_res = {f"test/{k}": np.mean([x[k] for x in self.test_step_outputs]) for k in ["PSNR", "SSIM", "LPIPS", "time"]}
        self.log_dict(avg_res, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()