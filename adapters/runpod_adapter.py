"""
Runpod Platform Adapter (for competition submission)
"""
import os
from .base import PlatformAdapter
from configs.base import Config


class RunpodAdapter(PlatformAdapter):
    """
    Adapter for running on Runpod (competition environment).
    
    Usage:
        # In train_gpt.py (submission file)
        if __name__ == "__main__":
            adapter = RunpodAdapter(config)
            trainer = Trainer(config, adapter)
            trainer.train()
    """
    
    def _setup(self):
        """Runpod-specific setup"""
        # Set paths from environment or defaults
        self.config.data_path = self.get_data_path()
        self.config.tokenizer_path = self.get_tokenizer_path()
        self.config.output_path = self.get_output_path()
        
        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # Setup distributed if multi-GPU
        self.setup_distributed()
    
    def get_data_path(self) -> str:
        """Get data path from env or default"""
        return os.environ.get(
            "DATA_PATH", 
            "./data/datasets/fineweb10B_sp1024"
        )
    
    def get_tokenizer_path(self) -> str:
        """Get tokenizer path from env or default"""
        return os.environ.get(
            "TOKENIZER_PATH",
            "./data/tokenizers/fineweb_1024_bpe.model"
        )
    
    def get_output_path(self) -> str:
        """Get output path from env or default"""
        return os.environ.get("OUTPUT_PATH", "./output")
