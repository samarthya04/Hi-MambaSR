"""
Hi-MambaSR Model Selection Factory.
Manages the initialization of different architectural variants for research benchmarks.
"""

import os
import torch
from .model_config_imports import *

def model_selection(cfg, device):
    """
    Select and initialize the model based on the research configuration.
    
    Supported models:
    - Hi-MambaSR: The primary Hierarchical Swin-Mamba Latent Diffusion GAN.
    - SupResDiffGAN_without_adv: Ablation variant without adversarial loss.
    - SupResDiffGAN_simple_gan: Legacy variant for baseline comparison.
    """
    if cfg.model.name == "Hi-MambaSR":
        # The primary architecture for the paper
        return initialize_model(
            cfg, device, HiMambaSR, use_discriminator=True
        )

    elif cfg.model.name == "SupResDiffGAN_without_adv":
        return initialize_model(
            cfg, device, SupResDiffGAN_without_adv, use_discriminator=False
        )

    elif cfg.model.name == "SupResDiffGAN_simple_gan":
        return initialize_model(
            cfg, device, SupResDiffGAN_simple_gan, use_discriminator=True
        )

    else:
        raise ValueError(
            f"Model '{cfg.model.name}' not identified. "
            f"Supported research models: Hi-MambaSR, SupResDiffGAN_without_adv, SupResDiffGAN_simple_gan"
        )


def initialize_model(cfg, device, model_class, use_discriminator=True):
    """
    Helper to initialize components and assemble the final LightningModule.
    """
    
    # 1. Initialize the Latent Projector (VAE/TinyVAE)
    autoencoder = get_vae(cfg.autoencoder, device)

    # 2. Initialize the Adversarial Critic
    discriminator = (
        Discriminator_engine(
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels,
        ) if use_discriminator else None
    )

    # 3. Initialize the Denoising Backbone (Hybrid Swin-Mamba UNet)
    unet = HybridUNet_backbone(cfg.unet)

    # 4. Initialize the Diffusion Schedule
    diffusion = Diffusion_engine(
        timesteps=cfg.diffusion.timesteps,
        beta_type=cfg.diffusion.beta_type,
        posterior_type=cfg.diffusion.posterior_type,
    )

    # 5. Initialize Perceptual Loss Modules
    vgg_loss = None
    if cfg.use_perceptual_loss:
        if cfg.feature_extractor:
            # Multi-scale feature matching
            vgg_loss = FeatureExtractor_engine()
        else:
            # Standard single-layer VGG loss
            vgg_loss = VGGLoss_engine()

    # 6. Assemble the LightningModule
    model = model_class(
        ae=autoencoder,
        discriminator=discriminator,
        unet=unet,
        diffusion=diffusion,
        learning_rate=cfg.model.lr,
        alfa_perceptual=cfg.model.alfa_perceptual,
        alfa_adv=cfg.model.alfa_adv,
        vgg_loss=vgg_loss,
        optimizer_8bit=cfg.trainer.get('optimizer_8bit', False),
    )

    # 7. Loading Strategy (Supports both .pth weights and .ckpt training states)
    if cfg.model.load_model:
        m_path = cfg.model.load_model
        _, ext = os.path.splitext(m_path)
        
        print(f"Loading pre-trained state from {m_path}...")
        
        if ext == ".pth":
            # Direct state_dict load
            state_dict = torch.load(m_path, map_location=device)
            # Use strict=False because our modern architectural fixes (RMSNorm, InstanceNorm) 
            # have different parameter keys (e.g. no bias) compared to legacy checkpoints.
            model.load_state_dict(state_dict, strict=False)
        elif ext == ".ckpt":
            # Lightning checkpoint recovery
            model = model_class.load_from_checkpoint(
                m_path,
                map_location=device,
                strict=False,
                ae=autoencoder,
                discriminator=discriminator,
                unet=unet,
                diffusion=diffusion,
                learning_rate=cfg.model.lr,
                alfa_perceptual=cfg.model.alfa_perceptual,
                alfa_adv=cfg.model.alfa_adv,
                vgg_loss=vgg_loss,
                optimizer_8bit=cfg.trainer.get('optimizer_8bit', False),
            )
        else:
            raise ValueError(f"Unsupported weight format: {ext}")

    return model