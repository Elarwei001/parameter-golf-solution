"""
Core Training Logic (Platform-agnostic)

This module contains the main training loop that works across all platforms.
Platform-specific code is handled by adapters.
"""
import os
import time
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

from configs.base import Config
from models.latent_lm import LatentLM
from adapters.base import PlatformAdapter


@dataclass
class TrainState:
    """Training state for checkpointing"""
    step: int = 0
    best_val_loss: float = float('inf')
    total_tokens: int = 0
    start_time: float = 0.0


class Trainer:
    """
    Main trainer class.
    
    Usage:
        config = get_default_config()
        adapter = ModalAdapter(config)  # or RunpodAdapter, LocalAdapter
        trainer = Trainer(config, adapter)
        result = trainer.train()
    """
    
    def __init__(self, config: Config, adapter: PlatformAdapter):
        self.config = config
        self.adapter = adapter
        self.device = adapter.get_device()
        self.state = TrainState()
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # EMA model (optional)
        self.ema_model = None
        if config.training.use_ema:
            self.ema_model = self._build_ema_model()
        
        # Log model info
        n_params = self.model.count_parameters()
        adapter.log(f"Model parameters: {n_params:,}")
        adapter.log(f"Estimated FP16 size: {self.model.estimate_size(16):.2f} MB")
        adapter.log(f"Estimated 3-bit size: {self.model.estimate_size(3):.2f} MB")
    
    def _build_model(self) -> LatentLM:
        """Build and initialize model"""
        model = LatentLM(self.config.model)
        model = model.to(self.device)
        
        # Compile for speed (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            model = torch.compile(model)
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer"""
        tc = self.config.training
        
        # Separate weight decay for different param groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'ln' in name or 'bias' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': tc.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        if tc.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=tc.learning_rate,
                betas=(tc.beta1, tc.beta2),
            )
        elif tc.optimizer == "muon":
            # TODO: Implement Muon optimizer
            # For now, fall back to AdamW
            self.adapter.log("Warning: Muon not implemented, using AdamW")
            return torch.optim.AdamW(
                param_groups,
                lr=tc.learning_rate,
                betas=(tc.beta1, tc.beta2),
            )
        else:
            raise ValueError(f"Unknown optimizer: {tc.optimizer}")
    
    def _build_ema_model(self) -> LatentLM:
        """Build EMA model"""
        import copy
        ema = copy.deepcopy(self.model)
        for param in ema.parameters():
            param.requires_grad = False
        return ema
    
    def _update_ema(self):
        """Update EMA model weights"""
        if self.ema_model is None:
            return
        
        decay = self.config.training.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.lerp_(param, 1 - decay)
    
    def _get_lr(self, step: int) -> float:
        """Get learning rate with warmup and warmdown"""
        tc = self.config.training
        
        # Warmup
        if step < tc.warmup_steps:
            return tc.learning_rate * (step + 1) / tc.warmup_steps
        
        # Warmdown
        if tc.max_steps is not None and step > tc.max_steps - tc.warmdown_steps:
            remaining = tc.max_steps - step
            return tc.learning_rate * remaining / tc.warmdown_steps
        
        return tc.learning_rate
    
    def _set_lr(self, lr: float):
        """Set learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def train_step(self, batch: torch.Tensor) -> dict:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Forward + loss
        loss_dict = self.model.compute_loss(
            batch, 
            sigreg_weight=self.config.training.sigreg_weight
        )
        
        # Backward
        self.optimizer.zero_grad()
        loss_dict['loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        
        # EMA update
        self._update_ema()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Evaluate on validation set"""
        model = self.ema_model if self.ema_model is not None else self.model
        model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            loss_dict = model.compute_loss(batch, sigreg_weight=0.0)
            
            batch_tokens = batch.numel()
            total_loss += loss_dict['ce_loss'].item() * batch_tokens
            total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens
        
        # Bits per byte (BPB)
        # BPB = loss / ln(2) for cross-entropy loss
        bpb = avg_loss / 0.6931  # ln(2) ≈ 0.6931
        
        return {
            'val_loss': avg_loss,
            'val_bpb': bpb,
            'val_ppl': torch.exp(torch.tensor(avg_loss)).item(),
        }
    
    def train(self, train_dataloader, val_dataloader=None) -> dict:
        """
        Main training loop.
        
        Returns:
            dict with final metrics
        """
        tc = self.config.training
        self.state.start_time = time.time()
        
        self.adapter.log("Starting training...")
        self.adapter.log(f"Max wallclock: {tc.max_wallclock_seconds}s")
        
        step = 0
        for batch in train_dataloader:
            # Check time limit
            elapsed = time.time() - self.state.start_time
            if elapsed >= tc.max_wallclock_seconds:
                self.adapter.log(f"Time limit reached ({elapsed:.1f}s)")
                break
            
            # Check step limit
            if tc.max_steps is not None and step >= tc.max_steps:
                self.adapter.log(f"Max steps reached ({step})")
                break
            
            # Update learning rate
            lr = self._get_lr(step)
            self._set_lr(lr)
            
            # Train step
            metrics = self.train_step(batch)
            
            # Logging
            if step % 50 == 0:
                self.adapter.log_metrics(step, {
                    'loss': metrics['loss'],
                    'ce': metrics['ce_loss'],
                    'sig': metrics['sigreg_loss'],
                    'lr': lr,
                })
            
            step += 1
        
        self.state.step = step
        
        # Final evaluation
        result = {'step': step, 'train_time': time.time() - self.state.start_time}
        
        if val_dataloader is not None:
            self.adapter.log("Running final evaluation...")
            val_metrics = self.evaluate(val_dataloader)
            result.update(val_metrics)
            self.adapter.log(f"Final val_bpb: {val_metrics['val_bpb']:.4f}")
        
        return result
    
    def save(self, path: str):
        """Save model checkpoint"""
        model = self.ema_model if self.ema_model is not None else self.model
        self.adapter.save_checkpoint(model, path, {
            'step': self.state.step,
            'config': self.config,
        })
        self.adapter.log(f"Saved checkpoint to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        metadata = self.adapter.load_checkpoint(self.model, path)
        self.state.step = metadata.get('step', 0)
        self.adapter.log(f"Loaded checkpoint from {path}")
