import torch
import wandb

def log_gpu_stats():
    """Log GPU memory usage to W&B"""
    if torch.cuda.is_available():
        wandb.log({
            "gpu/memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
            "gpu/memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
            "gpu/utilization": torch.cuda.utilization(0)
        })