"""
Platform Adapter Base Class
"""
import os
import torch
from abc import ABC, abstractmethod
from typing import Optional
from configs.base import Config


class PlatformAdapter(ABC):
    """
    Abstract base class for platform adapters.
    Handles differences between Modal, Runpod, and local environments.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Platform-specific setup"""
        pass
    
    @abstractmethod
    def get_data_path(self) -> str:
        """Get path to training data"""
        pass
    
    @abstractmethod
    def get_tokenizer_path(self) -> str:
        """Get path to tokenizer"""
        pass
    
    @abstractmethod
    def get_output_path(self) -> str:
        """Get path for outputs (checkpoints, logs)"""
        pass
    
    def get_device(self) -> torch.device:
        """Get compute device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def get_world_size(self) -> int:
        """Get number of distributed processes"""
        return int(os.environ.get("WORLD_SIZE", 1))
    
    def get_rank(self) -> int:
        """Get current process rank"""
        return int(os.environ.get("RANK", 0))
    
    def get_local_rank(self) -> int:
        """Get local process rank (for multi-GPU per node)"""
        return int(os.environ.get("LOCAL_RANK", 0))
    
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.get_rank() == 0
    
    def setup_distributed(self):
        """Setup distributed training if needed"""
        if self.get_world_size() > 1:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.get_local_rank())
    
    def cleanup_distributed(self):
        """Cleanup distributed training"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    
    def save_checkpoint(self, model: torch.nn.Module, path: str, metadata: dict = None):
        """Save model checkpoint"""
        if self.is_main_process():
            state = {
                "model": model.state_dict(),
                "metadata": metadata or {}
            }
            torch.save(state, path)
    
    def load_checkpoint(self, model: torch.nn.Module, path: str) -> dict:
        """Load model checkpoint"""
        state = torch.load(path, map_location=self.get_device())
        model.load_state_dict(state["model"])
        return state.get("metadata", {})
    
    def log(self, message: str):
        """Log message (only on main process)"""
        if self.is_main_process():
            print(message)
    
    def log_metrics(self, step: int, metrics: dict):
        """Log training metrics"""
        if self.is_main_process():
            metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"step {step} | {metrics_str}")
