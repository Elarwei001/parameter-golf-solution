"""
Modal Platform Adapter
"""
import os
from .base import PlatformAdapter
from configs.base import Config


class ModalAdapter(PlatformAdapter):
    """
    Adapter for running on Modal.
    
    Usage:
        @modal.function(gpu="A100")
        def train():
            adapter = ModalAdapter(config)
            trainer = Trainer(config, adapter)
            trainer.train()
    """
    
    # Modal volume paths
    DATA_VOLUME = "/data"
    OUTPUT_VOLUME = "/output"
    
    def _setup(self):
        """Modal-specific setup"""
        os.environ["MODAL_ENVIRONMENT"] = "1"
        
        # Set paths
        self.config.data_path = self.get_data_path()
        self.config.tokenizer_path = self.get_tokenizer_path()
        self.config.output_path = self.get_output_path()
        
        # Create output directory
        os.makedirs(self.config.output_path, exist_ok=True)
    
    def get_data_path(self) -> str:
        """Modal volume data path"""
        return os.path.join(self.DATA_VOLUME, "fineweb10B_sp1024")
    
    def get_tokenizer_path(self) -> str:
        """Modal volume tokenizer path"""
        return os.path.join(self.DATA_VOLUME, "tokenizers", "fineweb_1024_bpe.model")
    
    def get_output_path(self) -> str:
        """Modal volume output path"""
        return self.OUTPUT_VOLUME
    
    @staticmethod
    def create_image():
        """
        Create Modal image with dependencies.
        
        Usage in modal_runner.py:
            image = ModalAdapter.create_image()
            app = modal.App("parameter-golf", image=image)
        """
        import modal
        
        return (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install([
                "torch>=2.0",
                "numpy",
                "sentencepiece",
                "huggingface-hub",
                "datasets",
                "tqdm",
                "zstandard",
            ])
        )
    
    @staticmethod
    def create_data_volume():
        """Create Modal volume for data"""
        import modal
        return modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
    
    @staticmethod
    def create_output_volume():
        """Create Modal volume for outputs"""
        import modal
        return modal.Volume.from_name("parameter-golf-output", create_if_missing=True)
