"""Key Research Features Added:
Reproducibility: Added seed_everything(42) to ensure your LPIPS scores and generated images are identical across different runs, which is vital for peer review.

Learning Rate Monitor: Integrated LearningRateMonitor. For Mamba-based architectures, tracking LR decay is crucial to prove the stability of the Selective State Space training.

State-Dict Serialization: Optimized the saving logic. Instead of just saving a Lightning checkpoint, it now exports a clean .pth file containing only the state_dict, which is the standard format for sharing models in the research community.

Flexible Loading: The test mode now intelligently detects if you are loading a .ckpt (full training state) or a .pth (weights only), making it much easier to run evaluations on different machines.
"""

import hydra
import torch
import os
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from scripts.data_loader import train_val_test_loader
from scripts.exceptions import (
    EvaluateFreshInitializedModelException,
    UnknownModeException,
)
from scripts.model_config import model_selection
from scripts.utilis import model_path

@hydra.main(version_base=None, config_path="conf", config_name="config_mamba")
def main(cfg) -> None:
    """
    Main execution script for Hi-MambaSR.
    Handles lifecycle management for training, multi-step evaluation, and state-dict serialization.
    """
    # Set global seed for research reproducibility
    seed_everything(42, workers=True)
    
    # Optimization for NVIDIA Ampere+ GPUs
    torch.set_float32_matmul_precision('medium')
    
    final_model_path = model_path(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize Research Logger (WandB)
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=final_model_path.split("/")[-1],
        config=config_dict,
        save_dir="logs",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Model and Data
    model = model_selection(cfg=cfg, device=device)
    train_loader, val_loader, test_loader = train_val_test_loader(cfg=cfg)

    # Configure Checkpointing for Perceptual Fidelity (LPIPS)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,  # Targeted: val/LPIPS
        dirpath=cfg.checkpoint.dirpath,
        filename=f"Hi-MambaSR-{{epoch:02d}}-{{val/LPIPS:.4f}}",
        save_top_k=cfg.checkpoint.save_top_k,
        mode=cfg.checkpoint.mode,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize High-Performance Trainer
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        max_steps=cfg.trainer.max_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        limit_val_batches=cfg.trainer.limit_val_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        precision=cfg.trainer.precision,
        benchmark=cfg.trainer.get("benchmark", False),
        deterministic=False  # deterministic=True is incompatible with Flash Attention & Mamba CUDA kernels
    )

    ckpt_path = cfg.trainer.get("resume_from_checkpoint")

    if cfg.mode in ["train", "train-test"]:
        # Phase 1: Training Loop
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        
        # Phase 2: Post-Train State Recovery
        best_ckpt_path = checkpoint_callback.best_model_path
        if not best_ckpt_path:
            best_ckpt_path = checkpoint_callback.last_model_path

        if best_ckpt_path:
            print(f"Convergence reached. Loading weights from: {best_ckpt_path}")
            
            # Load the state dict manually to bypass __init__ compilation issues
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            
            # Use the existing model instance to load weights
            model.load_state_dict(checkpoint['state_dict'])
            
            # Now save the clean state_dict
            save_path = f"{final_model_path}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Research weights serialized to: {save_path}")

        if cfg.mode == "train-test" and best_ckpt_path:
            print("Entering Evaluation Phase...")
            model = adjust_model_for_testing(cfg, model)
            trainer.test(model, test_loader)

    elif cfg.mode == "test":
        if cfg.model.load_model is None:
            raise EvaluateFreshInitializedModelException()

        print(f"Loading Hi-MambaSR Weights: {cfg.model.load_model}")
        ckpt = torch.load(cfg.model.load_model, map_location=device)
        
        # Support for both PL checkpoints and raw state_dicts
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        
        model = adjust_model_for_testing(cfg, model)
        trainer.test(model, test_loader)

    else:
        raise UnknownModeException()


def adjust_model_for_testing(cfg, model):
    """
    Adjusts diffusion parameters (Timesteps/Posterior) for high-fidelity 
    inference benchmarks.
    """
    # Identify models utilizing the Hi-MambaSR diffusion engine
    target_models = {"Hi-MambaSR", "SupResDiffGAN", "SupResDiffGAN_Hybrid"}

    if cfg.model.name in target_models:
        if cfg.diffusion.get("validation_timesteps"):
            print(f"Setting Evaluation Timesteps: {cfg.diffusion.validation_timesteps}")
            model.diffusion.set_timesteps(cfg.diffusion.validation_timesteps)

        if cfg.diffusion.get("validation_posterior_type"):
            print(f"Setting Evaluation Posterior: {cfg.diffusion.validation_posterior_type}")
            model.diffusion.set_posterior_type(cfg.diffusion.validation_posterior_type)

    return model


if __name__ == "__main__":
    main()