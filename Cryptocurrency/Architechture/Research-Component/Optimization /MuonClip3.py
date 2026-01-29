import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def _newton_schulz_inv_sqrt(A: torch.Tensor, iters: int = 5, eps: float = 1e-6, 
                            damping: float = 0.0) -> torch.Tensor:
    """Compute (A + eps I)^(-1/2) via Newton–Schulz with optional damping.
    A must be symmetric PSD-ish (we add eps*I for safety).
    
    Args:
        A: Symmetric positive semi-definite matrix
        iters: Number of Newton-Schulz iterations
        eps: Regularization epsilon
        damping: Damping factor λ ∈ [0, 0.3] for ill-conditioned matrices
    """
    # Compute in fp32 for stability under AMP; cast back at end
    compute_dtype = torch.float32 if A.dtype in (torch.float16, torch.bfloat16) else A.dtype
    A32 = A.to(compute_dtype)
    fro = A32.norm(p='fro')
    if not torch.isfinite(fro) or fro.item() == 0.0:
        return torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    # Scale A for stability
    Y = A32 / fro
    I32 = torch.eye(A.shape[0], device=A.device, dtype=compute_dtype)
    Z = I32.clone()
    
    # Coupled Newton-Schulz iteration with optional damping
    for _ in range(iters):
        if damping > 0:
            T = 0.5 * ((3.0 - damping) * I32 - Z @ Y)
        else:
            T = 0.5 * (3.0 * I32 - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    
    # Rescale Z to be A^(-1/2)
    Z = Z / (fro.sqrt() + eps)
    return Z.to(A.dtype)


@torch.no_grad()
def muon_like_update(W: torch.Tensor, G: torch.Tensor, ns_iters: int = 5, lr: float = 1e-3,
                     wd: float = 0.0, momentum_buf: Optional[torch.Tensor] = None, beta: float = 0.9,
                     mu_scaling: bool = True, ns_damping: float = 0.0,
                     adaptive_damping: bool = False, adaptive_scaling: bool = False,
                     debug: bool = False) -> Optional[torch.Tensor]:
    """Muon-style update that orthogonalizes the gradient step for matrices.
    - W: [out, in] parameter
    - G: dL/dW with same shape
    
    Key principles (correct Muon implementation):
    1. Choose smaller Gram matrix based on dimensions (efficiency)
    2. Orthogonalize raw gradient FIRST (preserves orthogonality properties)
    3. Apply momentum AFTER in orthogonalized space (maintains stability)
    4. Apply μ-scaling for geometry-aware step sizes
    
    Returns:
        momentum_buf (updated), diagnostics dict (if debug=True)
    """
    # Short-circuit on empty/zero/NaN grads (still apply WD)
    if not torch.isfinite(G).all() or G.abs().max() == 0:
        if wd != 0.0:
            W.mul_(1.0 - lr * wd)
        return momentum_buf
    
    out_dim, in_dim = G.shape
    diagnostics = {}
    
    # Optional: compute condition number for adaptive damping
    effective_damping = ns_damping
    if adaptive_damping or debug:
        try:
            with torch.no_grad():
                cond = torch.linalg.cond(G.float())
                cond = torch.nan_to_num(cond, nan=1.0, posinf=1e6).item()
                diagnostics['cond'] = cond
                
                if adaptive_damping:
                    # λ = min(0.3, max(0.0, 0.1 * log10(cond + 1)))
                    effective_damping = min(0.3, max(0.0, 0.1 * math.log10(cond + 1)))
                    diagnostics['adaptive_damping'] = effective_damping
        except:
            cond = 1.0
    
    # Choose the smaller Gram matrix for computational efficiency
    # This is critical for performance and numerical stability
    try:
        if out_dim <= in_dim:
            # Use G @ G.T which is [out, out] - Left Preconditioner
            A = G @ G.T
            inv_sqrt = _newton_schulz_inv_sqrt(
                A + 1e-6 * torch.eye(A.shape[0], device=G.device, dtype=G.dtype),
                iters=ns_iters,
                damping=effective_damping
            )
            U = inv_sqrt @ G  # Orthogonalized gradient [out, out] @ [out, in] -> [out, in]
        else:
            # Use G.T @ G which is [in, in] - Right Preconditioner
            B = G.T @ G
            inv_sqrt = _newton_schulz_inv_sqrt(
                B + 1e-6 * torch.eye(B.shape[0], device=G.device, dtype=G.dtype),
                iters=ns_iters,
                damping=effective_damping
            )
            U = G @ inv_sqrt  # Orthogonalized gradient [out, in] @ [in, in] -> [out, in]
    except RuntimeError as e:
        # Precision fallback: if Newton-Schulz fails, use identity (SGD-like step)
        if debug:
            print(f"[Muon] Newton-Schulz failed, falling back to identity: {e}")
        U = G
        diagnostics['fallback'] = True
    
    # Apply momentum AFTER orthogonalization (CRITICAL: maintains orthogonality)
    # This ensures momentum operates in the orthogonalized space
    # and preserves the geometric properties that make Muon effective
    if momentum_buf is not None:
        momentum_buf.mul_(beta).add_(U, alpha=1.0)
        U_eff = momentum_buf
    else:
        U_eff = U
    
    # μ-scaling: geometry-aware step size (larger layers step slower)
    if mu_scaling:
        if adaptive_scaling:
            # Normalize by gradient norm for very deep transformers
            grad_norm = G.norm() + 1e-8
            scale = 0.2 * math.sqrt(max(out_dim, in_dim)) / grad_norm
        else:
            scale = 0.2 * math.sqrt(max(out_dim, in_dim))
        effective_lr = lr * scale
        diagnostics['scale'] = scale
    else:
        effective_lr = lr
        diagnostics['scale'] = 1.0
    
    # Decoupled weight decay (AdamW style)
    if wd != 0.0:
        W.mul_(1.0 - lr * wd)
    
    # Apply update (no .data needed inside no_grad)
    W.add_(U_eff, alpha=-effective_lr)
    
    # Optional diagnostic logging
    if debug:
        step_norm = U_eff.norm().item()
        diagnostics.update({
            'shape': G.shape,
            'step_norm': step_norm,
            'effective_lr': effective_lr
        })
        print(f"[Muon] shape={G.shape}, cond={diagnostics.get('cond', 0.0):.2e}, "
              f"step_norm={step_norm:.4f}, scale={diagnostics['scale']:.3f}, "
              f"damping={effective_damping:.3f}")
    
    return momentum_buf if not debug else (momentum_buf, diagnostics)


def _per_head_views(linear: nn.Linear, num_heads: int) -> List[torch.Tensor]:
    """Get a list of per-head weight views from a combined projection matrix."""
    W = linear.weight  # [out, in]
    d_out, _ = W.shape
    assert d_out % num_heads == 0, "out_features must be divisible by num_heads"
    d_head = d_out // num_heads
    # Return a list of tensors, each being a view of a single head's weights
    return [W.narrow(0, h * d_head, d_head) for h in range(num_heads)]


def _spectral_norm_bound(W: torch.Tensor, power_iters: int = 8) -> float:
    """Estimate spectral norm (max singular value) via power iteration."""
    if W.numel() == 0:
        return 0.0
    device, dtype = W.device, W.dtype
    # Start with a random vector in the input space
    v = torch.randn((W.shape[1],), device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)
    # Power iteration (no autograd graph)
    with torch.no_grad():
        for _ in range(power_iters):
            u = W @ v
            u = u / (u.norm() + 1e-12)
            v = W.T @ u
            v = v / (v.norm() + 1e-12)
    # The spectral norm is the norm of (W @ v)
    return float((W @ v).norm().item())


@dataclass
class QKClipConfig:
    """Configuration for QK-Clipping."""
    threshold: float = 128.0  # tau: max allowed (s_q * s_k) product
    alpha: float = 0.5        # split scaling: Wq *= eta^alpha, Wk *= eta^(1-alpha)
    power_iters: int = 8      # iterations for spectral norm estimation
    per_head_scaling: bool = False  # scale threshold by sqrt(d_head)


@torch.no_grad()
def qk_clip_linear_pair(q_proj: nn.Linear, k_proj: nn.Linear, num_heads: int, cfg: QKClipConfig):
    """Apply QK-Clipping to a pair of Q/K projection layers."""
    q_heads = _per_head_views(q_proj, num_heads)
    k_heads = _per_head_views(k_proj, num_heads)
    
    for Wq_h, Wk_h in zip(q_heads, k_heads):
        # Estimate spectral norms for this head
        sq = _spectral_norm_bound(Wq_h, cfg.power_iters)
        sk = _spectral_norm_bound(Wk_h, cfg.power_iters)
        
        prod = sq * sk
        
        # Optional per-head threshold scaling
        if cfg.per_head_scaling:
            d_head = Wq_h.shape[0]
            threshold = cfg.threshold * math.sqrt(d_head)
        else:
            threshold = cfg.threshold
        
        # If product exceeds threshold, rescale
        if prod > threshold and math.isfinite(prod) and prod > 0.0:
            eta = threshold / prod  # scaling factor < 1
            Wq_h.mul_(eta ** cfg.alpha)
            Wk_h.mul_(eta ** (1.0 - cfg.alpha))

class MHLA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_latents: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.num_latents = num_latents

        # Learned latent array
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model) / math.sqrt(d_model))

        # Phase A: L <- Attn(L, X) (Latents query Tokens)
        self.qA = nn.Linear(d_model, d_model, bias=False)   # queries from latent
        self.kA = nn.Linear(d_model, d_model, bias=False)   # keys from tokens X
        self.vA = nn.Linear(d_model, d_model, bias=False)   # values from tokens X
        self.oA = nn.Linear(d_model, d_model, bias=False)

        # Phase B: X <- Attn(X, L) (Tokens query Latents)
        self.qB = nn.Linear(d_model, d_model, bias=False)   # queries from tokens X
        self.kB = nn.Linear(d_model, d_model, bias=False)   # keys from latent
        self.vB = nn.Linear(d_model, d_model, bias=False)   # values from latent
        self.oB = nn.Linear(d_model, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.d_head)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, H, T, Dh]
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Dh] -> [B, T, C]
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # replicate latents across batch
        L = self.latents.expand(B, -1, -1)  # [B, L, C]

        # ---- Phase A: L' = Attn(L, X) ----
        qA = self._split_heads(self.qA(L))          # [B,H,L,Dh]
        kA = self._split_heads(self.kA(x))          # [B,H,T,Dh]
        vA = self._split_heads(self.vA(x))          # [B,H,T,Dh]
        attA = (qA @ kA.transpose(-2, -1)) * self.scale  # [B,H,L,T]
        attA = attA.softmax(dim=-1)
        LA = attA @ vA                               # [B,H,L,Dh]
        LA = self.oA(self._merge_heads(LA))          # [B,L,C]

        # Residual on latent side (prevents drift)
        L = L + LA

        # ---- Phase B: X' = Attn(X, L) ----
        qB = self._split_heads(self.qB(x))          # [B,H,T,Dh]
        kB = self._split_heads(self.kB(L))          # [B,H,L,Dh]
        vB = self._split_heads(self.vB(L))          # [B,H,L,Dh]
        attB = (qB @ kB.transpose(-2, -1)) * self.scale  # [B,H,T,L]
        attB = attB.softmax(dim=-1)
        XB = attB @ vB                               # [B,H,T,Dh]
        XB = self.oB(self._merge_heads(XB))          # [B,T,C]
        return XB

    # expose q/k pairs for both phases to QK-Clip
    def iter_qk_pairs(self) -> Iterable[Tuple[nn.Linear, nn.Linear, int]]:
        yield self.qA, self.kA, self.num_heads  # Latent<-Tokens
        yield self.qB, self.kB, self.num_heads  # Tokens<-Latent


class LatentBlock(nn.Module):
    """Standard Transformer block using MHLA."""
    def __init__(self, d_model: int, num_heads: int, num_latents: int, mlp_ratio: int = 4, pdrop: float = 0.1):
        super().__init__()
        self.attn = MHLA(d_model, num_heads, num_latents)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model, bias=False),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model, bias=False),
        )
        self.drop = nn.Dropout(pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class TinyLatentTransformer(nn.Module):
    """A small transformer model using LatentBlocks."""
    def __init__(self, vocab: int, d_model: int = 256, num_heads: int = 8, depth: int = 4,
                 max_len: int = 128, num_latents: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.blocks = nn.ModuleList([LatentBlock(d_model, num_heads, num_latents) for _ in range(depth)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.max_len = max_len
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        assert T <= self.max_len, f"Input sequence length {T} exceeds max_len {self.max_len}"
        h = self.emb(x) + self.pos[:, :T, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)

    # expose all q/k projection pairs to optimizer for QK-Clip
    def iter_qk_pairs(self) -> Iterable[Tuple[nn.Linear, nn.Linear, int]]:
        for blk in self.blocks:
            for pair in blk.attn.iter_qk_pairs():
                yield pair

class MuonClip(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, ns_iters=5, momentum=0.9,
                 qk_clip: Optional[QKClipConfig] = QKClipConfig(),
                 model_with_qk=None,
                 mu_scaling: bool = True,
                 ns_damping: float = 0.0,
                 adaptive_damping: bool = False,
                 adaptive_scaling: bool = False,
                 momentum_warmup_steps: int = 0,
                 reorth_interval: int = 0,
                 nesterov_1d: bool = False,
                 grad_clip_norm: Optional[float] = None,
                 fsdp_compatible: bool = False,
                 logging_callback: Optional[callable] = None,
                 debug: bool = False):
        """
        Optimizer combining Muon-style preconditioning for matrices and Momentum SGD for vectors.
        
        Args:
            params: iterable of parameters
            lr: learning rate
            weight_decay: decoupled weight decay (AdamW-style)
            ns_iters: iterations for Newton-Schulz inv_sqrt
            momentum: momentum factor (beta)
            qk_clip: (optional) config for QK-Clipping
            model_with_qk: (optional) model instance exposing .iter_qk_pairs() for QK-Clipping
            mu_scaling: enable μ-scaling (geometry-aware step sizes)
            ns_damping: Newton-Schulz damping λ ∈ [0, 0.3] for ill-conditioned matrices
            adaptive_damping: auto-adjust damping based on gradient condition number
            adaptive_scaling: normalize μ-scaling by gradient norm (for very deep nets)
            momentum_warmup_steps: steps to warm up momentum (0 = disabled)
            reorth_interval: re-orthogonalize momentum buffers every N steps (0 = disabled)
            nesterov_1d: use Nesterov momentum for 1D params
            grad_clip_norm: optional gradient norm clipping before step
            fsdp_compatible: skip params not local to rank (for FSDP/ZeRO)
            logging_callback: optional function(step, metrics_dict) for logging
            debug: enable diagnostic logging
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_damping < 0.0 or ns_damping > 0.5:
            raise ValueError(f"Invalid ns_damping value: {ns_damping}")
            
        defaults = dict(lr=lr, weight_decay=weight_decay, ns_iters=ns_iters, momentum=momentum)
        super().__init__(params, defaults)
        
        self._momentum = {}
        self.qk_cfg = qk_clip
        self.model = model_with_qk
        self.mu_scaling = mu_scaling
        self.ns_damping = ns_damping
        self.adaptive_damping = adaptive_damping
        self.adaptive_scaling = adaptive_scaling
        self.momentum_warmup_steps = momentum_warmup_steps
        self.reorth_interval = reorth_interval
        self.nesterov_1d = nesterov_1d
        self.grad_clip_norm = grad_clip_norm
        self.fsdp_compatible = fsdp_compatible
        self.logging_callback = logging_callback
        self.debug = debug
        self._step_count = 0
        self._metrics = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        self._metrics = {'step': self._step_count}
        
        # Optional gradient clipping before Muon step
        if self.grad_clip_norm is not None:
            # Collect all parameters with gradients
            params_with_grad = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
            if params_with_grad:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    params_with_grad, self.grad_clip_norm
                )
                self._metrics['grad_norm_pre_clip'] = total_norm.item()
        
        # Compute adaptive momentum with warmup
        qk_metrics = []
        
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            ns_iters = group['ns_iters']
            base_mom = group['momentum']
            
            # Momentum warmup: β_t = β * (1 - exp(-t/τ))
            if self.momentum_warmup_steps > 0:
                tau = self.momentum_warmup_steps
                mom = base_mom * (1.0 - math.exp(-self._step_count / tau))
                self._metrics['effective_momentum'] = mom
            else:
                mom = base_mom
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # FSDP compatibility: skip non-local parameters
                if self.fsdp_compatible and hasattr(p, '_is_sharded'):
                    if not p._is_sharded or not hasattr(p, '_local_shard'):
                        continue
                
                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("MuonClip does not support sparse gradients")

                # Get or initialize momentum buffer
                buf = self._momentum.get(p)
                if buf is None:
                    buf = torch.zeros_like(g)
                    self._momentum[p] = buf

                if g.ndim == 2:
                    # --- 2D Matrix Param ---
                    # Use Muon update: orthogonalize first, then momentum
                    result = muon_like_update(
                        p, g, 
                        ns_iters=ns_iters, 
                        lr=lr, 
                        wd=wd, 
                        momentum_buf=buf, 
                        beta=mom,
                        mu_scaling=self.mu_scaling,
                        ns_damping=self.ns_damping,
                        adaptive_damping=self.adaptive_damping,
                        adaptive_scaling=self.adaptive_scaling,
                        debug=self.debug
                    )
                    
                    # Handle debug return value (momentum_buf, diagnostics)
                    if self.debug and isinstance(result, tuple):
                        buf, diag = result
                        # Aggregate diagnostics for logging
                        for k, v in diag.items():
                            if k not in self._metrics:
                                self._metrics[k] = []
                            self._metrics[k].append(v)
                    
                    # Optional: Re-orthogonalize momentum buffer periodically
                    if self.reorth_interval > 0 and self._step_count % self.reorth_interval == 0:
                        if self.debug:
                            print(f"[Muon] Re-orthogonalizing momentum buffer at step {self._step_count}")
                        # Re-project momentum onto orthogonal subspace
                        try:
                            muon_like_update(
                                buf, buf,
                                ns_iters=ns_iters,
                                lr=1.0,  # Just orthogonalize, don't step
                                wd=0.0,
                                momentum_buf=None,
                                beta=0.0,
                                mu_scaling=False,
                                ns_damping=self.ns_damping,
                                adaptive_damping=False,
                                adaptive_scaling=False,
                                debug=False
                            )
                        except:
                            pass  # If re-orth fails, keep existing buffer
                else:
                    # --- 1D Vector/Other Params (e.g., bias, LayerNorm) ---
                    # Use momentum SGD with decoupled WD (optionally Nesterov)
                    
                    # Decoupled weight decay
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    
                    # Momentum update
                    buf.mul_(mom).add_(g)
                    
                    if self.nesterov_1d:
                        # Nesterov momentum: step with look-ahead
                        p.add_(buf, alpha=-lr * mom)
                        p.add_(g, alpha=-lr)
                    else:
                        # Standard momentum
                        p.add_(buf, alpha=-lr)

        # After all parameter updates, apply the optional QK-Clip pass
        if self.model is not None and self.qk_cfg is not None:
            for q_proj, k_proj, num_heads in self.model.iter_qk_pairs():
                # Store pre-clip spectral norms for logging
                if self.debug or self.logging_callback:
                    q_heads = _per_head_views(q_proj, num_heads)
                    k_heads = _per_head_views(k_proj, num_heads)
                    for i, (Wq_h, Wk_h) in enumerate(zip(q_heads, k_heads)):
                        sq = _spectral_norm_bound(Wq_h, self.qk_cfg.power_iters)
                        sk = _spectral_norm_bound(Wk_h, self.qk_cfg.power_iters)
                        qk_metrics.append({
                            'head_idx': i,
                            'sq': sq,
                            'sk': sk,
                            'product': sq * sk
                        })
                
                qk_clip_linear_pair(q_proj, k_proj, num_heads, self.qk_cfg)
        
        # Store QK metrics
        if qk_metrics:
            self._metrics['qk_spectral_products'] = [m['product'] for m in qk_metrics]
            self._metrics['qk_max_product'] = max(m['product'] for m in qk_metrics)
            self._metrics['qk_mean_product'] = sum(m['product'] for m in qk_metrics) / len(qk_metrics)
        
        # Call logging callback if provided
        if self.logging_callback is not None:
            self.logging_callback(self._step_count, self._metrics)
        
        return loss
    
    def get_metrics(self) -> dict:
        """Return the most recent step metrics."""
        return self._metrics.copy()
    
    def state_dict(self):
        """Return optimizer state including step count."""
        state_dict = super().state_dict()
        state_dict['step_count'] = self._step_count
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load optimizer state including step count."""
        self._step_count = state_dict.pop('step_count', 0)
        super().load_state_dict(state_dict)
