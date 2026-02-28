"""Key Research Features Added:
Metric Cleaning: Automatically strips test/ and val/ prefixes from logged data to ensure that your CSV exports and WandB charts use clean, publication-ready labels (e.g., LPIPS instead of test/LPIPS).

Unique Export Tracking: The save_results_to_csv function now uses a counter to prevent overwriting results if you run multiple evaluations back-to-back.

Comprehensive Configuration Support: The logic handles mode: all, mode: steps, and mode: posterior dynamically, allowing you to generate the specific data needed for different tables in your paper (e.g., a sampling speed vs. quality table).

Flexible Weight Loading: Added logic to handle both raw .pth files and full .ckpt files, ensuring that the script works regardless of how you serialized the model during training.

Multi-GPU Inference Support: The script is fully compatible with Lightning's accelerator and devices configuration, allowing for high-speed evaluation on large test sets.
"""

import os
import warnings
import hydra
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from typing import List, Dict

from scripts.data_loader import train_val_test_loader
from scripts.exceptions import (
    EvaluateFreshInitializedModelException,
    UnknownModeException,
)
from scripts.model_config import model_selection
from scripts.utilis import model_path

def save_results_to_csv(results: List[Dict], filename: str) -> None:
    """
    Serializes evaluation benchmarks to a CSV file for research reporting.
    Ensures unique filenames to prevent overwriting previous experiment data.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    base, ext = os.path.splitext(filename)
    if ext.lower() != ".csv":
        filename = f"{base}.csv"

    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}.csv"
        counter += 1

    df = pd.DataFrame(results)
    df.to_csv(new_filename, index=False)
    print(f"Evaluation metrics successfully saved to: {new_filename}")

def log_visual_metrics_to_wandb(results: List[Dict], mode: str) -> None:
    """
    Logs comparative bar charts to WandB for qualitative performance analysis.
    """
    metrics = {}
    for result in results:
        m_name = result["metric"]
        # Generate x-axis labels based on the evaluation sweep mode
        if mode == "all":
            label = f"{result['posterior']}_steps_{result['step']}"
        else:
            label = str(result.get("step", result.get("posterior")))
        
        if m_name not in metrics:
            metrics[m_name] = []
        metrics[m_name].append([label, result["value"]])

    for m_name, data in metrics.items():
        table = wandb.Table(data=data, columns=["Configuration", m_name])
        wandb.log({f"eval/chart_{m_name}": wandb.plot.bar(table, "Configuration", m_name)})

def run_evaluation_suite(cfg, model, trainer: Trainer, test_loader) -> None:
    """
    Executes a comprehensive evaluation sweep across different diffusion 
    sampling trajectories (DDIM/DDPM) and timestep resolutions.
    """
    results = []
    eval_mode = cfg.evaluation.mode
    posteriors = cfg.evaluation.posteriors if eval_mode in ["posterior", "all"] else [model.diffusion.posterior_type]
    steps = cfg.evaluation.steps if eval_mode in ["steps", "all"] else [model.diffusion.timesteps]

    print(f"Starting Hi-MambaSR Evaluation Suite [Mode: {eval_mode}]")

    for posterior in posteriors:
        model.diffusion.set_posterior_type(posterior)
        for step_count in steps:
            print(f"Benchmarking Configuration: Posterior={posterior}, Steps={step_count}")
            model.diffusion.set_timesteps(step_count)
            
            # Execute standard Lightning test loop
            trainer.test(model, test_loader)
            
            # Capture and clean metrics for reporting
            metrics = trainer.callback_metrics
            for m_key, m_val in metrics.items():
                clean_name = m_key.replace("test/", "").replace("val/", "")
                results.append({
                    "model": "Hi-MambaSR",
                    "posterior": posterior,
                    "step": step_count,
                    "metric": clean_name,
                    "value": m_val.item() if torch.is_tensor(m_val) else m_val
                })

    # Qualitative Logging
    log_visual_metrics_to_wandb(results, eval_mode)

    # Persistence
    if cfg.evaluation.save_results:
        save_results_to_csv(results, cfg.evaluation.results_file)

@hydra.main(version_base=None, config_path="conf", config_name="config_mamba")
def main(cfg) -> None:
    """
    Inference and Benchmarking entry point for Hi-MambaSR.
    """
    # Set global seed for research reproducibility
    seed_everything(42, workers=True)
    
    # Precision configuration for high-performance inference
    torch.set_float32_matmul_precision('medium')
    
    if cfg.model.load_model is None:
        raise EvaluateFreshInitializedModelException("Model path must be specified for evaluation.")

    final_model_path = model_path(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize Research Logger
    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"Eval_{final_model_path.split('/')[-1]}",
        config=config_dict,
        save_dir="logs",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model with appropriate Hi-MambaSR architecture injection
    print(f"Initializing Hi-MambaSR architecture on {device}")
    model = model_selection(cfg=cfg, device=device)
    _, _, test_loader = train_val_test_loader(cfg=cfg)

    # Load Weights (Supports both raw state_dicts and Lightning checkpoints)
    print(f"Loading weights from: {cfg.model.load_model}")
    ckpt = torch.load(cfg.model.load_model, map_location=device)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Initialize Evaluator
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
        deterministic=False  # deterministic=True is incompatible with Flash Attention & Mamba CUDA kernels
    )   

    run_evaluation_suite(cfg, model, trainer, test_loader)

if __name__ == "__main__":
    main()