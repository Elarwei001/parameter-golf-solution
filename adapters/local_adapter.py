"""
Local Platform Adapter (for development and debugging)
"""
import os
from .base import PlatformAdapter
from configs.base import Config


class LocalAdapter(PlatformAdapter):
    """
    Adapter for local development and debugging.
    Works on CPU, MPS (Mac), or single GPU.
    
    Usage:
        adapter = LocalAdapter(config)
        trainer = Trainer(config, adapter)
        trainer.train()
    """
    
    def __init__(self, config: Config, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        super().__init__(config)
    
    def _setup(self):
        """Local setup"""
        self.config.data_path = self.get_data_path()
        self.config.tokenizer_path = self.get_tokenizer_path()
        self.config.output_path = self.get_output_path()
        
        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Log device info
        device = self.get_device()
        self.log(f"Using device: {device}")
        if device.type == "cuda":
            self.log(f"GPU: {torch.cuda.get_device_name()}")
    
    def get_data_path(self) -> str:
        """Local data path"""
        return os.path.join(self.base_path, "data", "datasets", "fineweb10B_sp1024")
    
    def get_tokenizer_path(self) -> str:
        """Local tokenizer path"""
        return os.path.join(self.base_path, "data", "tokenizers", "fineweb_1024_bpe.model")
    
    def get_output_path(self) -> str:
        """Local output path"""
        return os.path.join(self.base_path, "output")


# Import torch here to avoid issues when module is imported
import torch
