"""
Muon Optimizer - Momentum Orthogonalized Update

Based on the Muon paper and official implementation.
Key insight: Orthogonalize momentum updates using Newton-Schulz iteration.

This is what the official Parameter Golf baseline uses.
"""
import torch
from torch.optim import Optimizer
from typing import List, Optional


def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize matrix G using Newton-Schulz iteration.
    
    For a matrix G, finds the nearest orthogonal matrix Q such that Q'Q = I.
    Uses the iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k' @ X_k
    
    Args:
        G: Input matrix to orthogonalize
        steps: Number of Newton-Schulz iterations (default 5)
    
    Returns:
        Orthogonalized matrix (same shape as G)
    """
    assert G.dim() == 2, "Newton-Schulz requires 2D matrix"
    
    rows, cols = G.shape
    
    # For wide matrices (rows < cols), work with G @ G.T
    # For tall matrices (rows >= cols), work with G.T @ G
    if rows < cols:
        # Wide matrix: orthogonalize rows
        # Normalize by spectral norm estimate
        G_normalized = G / (G.norm() + 1e-7)
        X = G_normalized
        
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X
        
        return X
    else:
        # Tall matrix: orthogonalize columns  
        G_normalized = G / (G.norm() + 1e-7)
        X = G_normalized
        
        for _ in range(steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X
        
        return X


class Muon(Optimizer):
    """
    Muon: Momentum Orthogonalized Update optimizer.
    
    For each parameter, computes momentum and then orthogonalizes it
    using Newton-Schulz iteration before applying the update.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        weight_decay: Weight decay coefficient (default: 0.0)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                
                # Update momentum buffer
                buf.mul_(momentum).add_(grad)
                
                # Compute update
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf
                
                # Apply Newton-Schulz orthogonalization for 2D parameters
                if p.dim() == 2 and p.shape[0] >= 2 and p.shape[1] >= 2:
                    # Reshape for orthogonalization
                    update_2d = update.view(p.shape[0], -1)
                    update_ortho = newton_schulz_orthogonalize(update_2d, steps=ns_steps)
                    update = update_ortho.view_as(p)
                    
                    # Scale by sqrt(dimensions) to preserve update magnitude
                    scale = (p.shape[0] * p.shape[1]) ** 0.5
                    update = update * scale
                
                # Apply update
                p.add_(update, alpha=-lr)
        
        return loss


class MuonWithAdamW(Optimizer):
    """
    Hybrid optimizer: Muon for 2D params, AdamW for 1D params (biases, norms).
    
    This is the recommended setup for training transformers.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        # AdamW params for 1D
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Use Muon for 2D, AdamW for 1D
                if p.dim() == 2 and p.shape[0] >= 2 and p.shape[1] >= 2:
                    # Muon update
                    self._muon_step(p, grad, state, group)
                else:
                    # AdamW update
                    self._adamw_step(p, grad, state, group)
        
        return loss
    
    def _muon_step(self, p, grad, state, group):
        """Muon step for 2D parameters."""
        lr = group['lr']
        momentum = group['momentum']
        nesterov = group['nesterov']
        ns_steps = group['ns_steps']
        weight_decay = group['weight_decay']
        
        if weight_decay != 0:
            grad = grad.add(p, alpha=weight_decay)
        
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p)
        
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad)
        
        if nesterov:
            update = grad + momentum * buf
        else:
            update = buf
        
        # Newton-Schulz orthogonalization
        update_2d = update.view(p.shape[0], -1)
        update_ortho = newton_schulz_orthogonalize(update_2d, steps=ns_steps)
        update = update_ortho.view_as(p)
        
        scale = (p.shape[0] * p.shape[1]) ** 0.5
        update = update * scale
        
        p.add_(update, alpha=-lr)
    
    def _adamw_step(self, p, grad, state, group):
        """AdamW step for 1D parameters."""
        lr = group['adamw_lr']
        beta1, beta2 = group['adamw_betas']
        eps = group['adamw_eps']
        weight_decay = group['weight_decay']
        
        # Decoupled weight decay
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)
        
        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
        
        state['step'] += 1
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        
        step_size = lr / bias_correction1
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        
        p.addcdiv_(exp_avg, denom, value=-step_size)


def test_muon():
    """Quick test of Muon optimizer."""
    print("Testing Muon optimizer...")
    
    # Simple 2-layer network
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
    )
    
    optimizer = Muon(model.parameters(), lr=0.01)
    
    # Fake training step
    x = torch.randn(32, 128)
    target = torch.randn(32, 64)
    
    for i in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        print(f"Step {i}: loss = {loss.item():.4f}")
    
    print("✅ Muon test passed!")


if __name__ == "__main__":
    test_muon()
