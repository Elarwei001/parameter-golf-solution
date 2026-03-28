"""
Configuration for Parameter Golf
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Tokenizer
    vocab_size: int = 1024
    
    # Encoder
    embed_dim: int = 256
    latent_dim: int = 64
    
    # Predictor (main compute)
    n_layers: int = 8
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    
    # Decoder
    tie_weights: bool = True  # Tie decoder with encoder


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    seq_length: int = 1024
    batch_size: int = 64
    total_batch_tokens: int = 2 ** 19  # 512K tokens per step
    
    # Optimizer
    optimizer: str = "muon"  # "adamw" or "muon"
    learning_rate: float = 1e-3
    weight_decay: float = 0.04
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Schedule
    warmup_steps: int = 200
    warmdown_steps: int = 3500
    max_steps: Optional[int] = None
    max_wallclock_seconds: int = 300  # 5 min for dev, 600 for final
    
    # Regularization
    sigreg_weight: float = 0.1
    
    # Model averaging
    use_ema: bool = True
    ema_decay: float = 0.95


@dataclass
class QuantConfig:
    """Quantization configuration (TurboQuant)"""
    enabled: bool = True
    embed_bits: int = 4
    weight_bits: int = 3
    use_qjl: bool = True  # QJL error correction


@dataclass
class EvalConfig:
    """Evaluation configuration"""
    sliding_window: bool = True
    window_stride: int = 64
    use_ttt: bool = False  # Test-time training
    ttt_lr: float = 1e-4
    ttt_steps: int = 1


@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = None
    training: TrainingConfig = None
    quant: QuantConfig = None
    eval: EvalConfig = None
    
    # Paths (set by adapter)
    data_path: str = ""
    tokenizer_path: str = ""
    output_path: str = ""
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.quant is None:
            self.quant = QuantConfig()
        if self.eval is None:
            self.eval = EvalConfig()


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def get_debug_config() -> Config:
    """Get small config for local debugging"""
    config = Config()
    config.model.n_layers = 2
    config.model.latent_dim = 32
    config.training.seq_length = 256
    config.training.batch_size = 4
    config.training.max_steps = 10
    config.training.max_wallclock_seconds = 60
    return config
