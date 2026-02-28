"""
Hi-MambaSR: Hierarchical State-Space Refinement for Latent Diffusion Super-Resolution.
Ultra-Optimized for 6GB VRAM: Flash Attention + Activation Checkpointing + Fused Scanning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.unets import UNet2DModel
from mamba_ssm import Mamba
from typing import List, Union

class RMSNorm(nn.Module):
    """Root Mean Square Normalization for stabilizing State-Space models."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwinBlock(nn.Module):
    """
    Optimized Swin Transformer Block.
    Uses Flash Attention and Gradient Checkpointing to survive low VRAM environments.
    """
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, shift_size: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1),
        )

    def _attention_block(self, x, H, W):
        """Internal function for checkpointing attention mechanism."""
        B, Hp, Wp, C = x.shape
        # Cyclic Shift
        if self.shift_size > 0:
            actual_shift = self.shift_size % self.window_size
            x = torch.roll(x, shifts=(-actual_shift, -actual_shift), dims=(1, 2))

        # Partition
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        # QKV Split
        qkv = self.qkv(windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Memory Efficient Scaled Dot Product Attention (Flash Attention)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)
        attn_out = self.proj(attn_out)

        # Reverse
        x = attn_out.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(actual_shift, actual_shift), dims=(1, 2))
        
        return x[:, :H, :W, :].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1) # To B, H, W, C
        x = self.norm1(x)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        # Apply Gradient Checkpointing to the memory-heavy attention part
        # This saves significant VRAM by recomputing during backward pass
        x = checkpoint(self._attention_block, x, H, W, use_reentrant=False)

        x = x.permute(0, 3, 1, 2) + shortcut 
        
        # FFN
        res = x
        x = self.norm2(x.permute(0, 2, 3, 1))
        x = res + self.mlp(x).permute(0, 3, 1, 2)
        return x

class MultiHeadSelectiveScan(nn.Module):
    """Fused Bi-directional Mamba."""
    def __init__(self, dim: int, scale_factor: int = 4):
        super().__init__()
        self.mamba = Mamba(d_model=dim, d_state=32, d_conv=4, expand=1)
        self.scale_modulator = nn.Parameter(torch.ones(1) * (1.0 / scale_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(B, -1, C)
        
        # Fused bi-directional scanning
        fwd_out = self.mamba(x_flat)
        bwd_out = self.mamba(x_flat.flip(dims=[1])).flip(dims=[1])
        
        out = (fwd_out + bwd_out) * self.scale_modulator
        return out.view(B, H, W, C).permute(0, 3, 1, 2)

class HiMambaBottleneck(nn.Module):
    """Optimized Mamba Bottleneck with Parallel Spatial Gating."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.global_ssm = MultiHeadSelectiveScan(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.res_scale = nn.Parameter(torch.zeros(1)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Checkpoint the SSM block as it is the VRAM peak in the bottleneck
        global_feat = checkpoint(self.global_ssm, x_norm, use_reentrant=False)
        local_feat = torch.sigmoid(self.dwconv(x_norm)) * x_norm
        
        return shortcut + self.res_scale * (global_feat + local_feat)

class HybridUNet(nn.Module):
    def __init__(self, channels: List[int] = [64, 96, 128, 256]):
        super().__init__()
        # Standard UNet backbone with attention only at the peak latent depth
        self.backbone = UNet2DModel(
            in_channels=8, out_channels=4,
            block_out_channels=channels,
            layers_per_block=2,
            add_attention=[False, False, False, True], 
        )
        # IMPORTANT: Separate SwinBlock instances for encoder and decoder.
        # Sharing weights (as a single instance) halves capacity at the deepest level.
        self.swin_down = SwinBlock(dim=channels[-1])
        self.swin_up = SwinBlock(dim=channels[-1])
        self.mamba_bottleneck = HiMambaBottleneck(dim=channels[-1])
        self._inject_custom_logic()
        
        # Enable backbone level checkpointing
        if hasattr(self.backbone, "enable_gradient_checkpointing"):
            self.backbone.enable_gradient_checkpointing()

    def _inject_custom_logic(self):
        class MambaMidWrapper(nn.Module):
            def __init__(self, mid, mamba):
                super().__init__()
                self.mid, self.mamba = mid, mamba
            def forward(self, sample, emb):
                # Using checkpointing for the mid-block fusion
                return self.mamba(self.mid(sample, emb))

        class SwinBlockWrapper(nn.Module):
            def __init__(self, block, swin):
                super().__init__()
                self.block, self.swin = block, swin
            def forward(self, hidden_states, temb=None, **kwargs):
                out, res = self.block(hidden_states=hidden_states, temb=temb, **kwargs)
                return self.swin(out), res
            def __getattr__(self, name):
                try: return super().__getattr__(name)
                except AttributeError: return getattr(self.block, name)

        class SwinUpBlockWrapper(nn.Module):
            def __init__(self, block, swin):
                super().__init__()
                self.block, self.swin = block, swin
            def forward(self, hidden_states, res_hidden_states_tuple, temb=None, **kwargs):
                out = self.block(hidden_states, res_hidden_states_tuple, temb, **kwargs)
                return self.swin(out)
            def __getattr__(self, name):
                try: return super().__getattr__(name)
                except AttributeError: return getattr(self.block, name)

        self.backbone.mid_block = MambaMidWrapper(self.backbone.mid_block, self.mamba_bottleneck)
        self.backbone.down_blocks[-1] = SwinBlockWrapper(self.backbone.down_blocks[-1], self.swin_down)
        self.backbone.up_blocks[0] = SwinUpBlockWrapper(self.backbone.up_blocks[0], self.swin_up)

    def forward(self, lr_latent, x_t, t):
        return self.backbone(torch.cat([x_t, lr_latent], dim=1), timestep=t).sample

class UNet(nn.Module):
    def __init__(self, cfg_unet: Union[List[int], object]):
        super().__init__()
        channels = list(cfg_unet) if hasattr(cfg_unet, '__iter__') else [64, 96, 128, 256]
        self.model = HybridUNet(channels=channels)

    def forward(self, lr_latent, x_t, t):
        return self.model(lr_latent, x_t, t)