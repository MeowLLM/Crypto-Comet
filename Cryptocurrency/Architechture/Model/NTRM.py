import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# RMSNorm
# =====================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., D)
        dim = x.size(-1)
        norm = x.norm(dim=-1, keepdim=True) * (dim ** -0.5)
        return self.weight * x / (norm + self.eps)


# =====================
# Multi-Head Self-Attention
# =====================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, causal=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B,T,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(t):
            # [B,T,D] -> [B,H,T,Hd]
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = scores.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B,H,T,Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# =====================
# Simple token-wise MoE
# =====================

class SimpleMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x):
        # x: [B,T,D]
        B, T, D = x.shape
        logits = self.router(x)              # [B,T,E]
        weights = F.softmax(logits, dim=-1)  # [B,T,E]

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # each [B,T,D]

        # [B,T,E,D]
        stacked = torch.stack(expert_outputs, dim=2)
        weights_exp = weights.unsqueeze(-1)   # [B,T,E,1]
        out = (weights_exp * stacked).sum(dim=2)  # [B,T,D]
        return out


# =====================
# Selector gate (MoE vs Attention)
# =====================

class SelectorGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B,T,D]
        g = torch.sigmoid(self.lin(x))  # [B,T,1] in (0,1)
        return g


# =====================
# Tiny Recursive MoE-Attention Model
# =====================

class TinyRecursiveMoEAttnModel(nn.Module):
    """
    Input  --> embedding
            --> RMSNorm
            --> (latent + residual from input)
            --> (+ prediction feedback)
            --> Selector -> MoE / Attention mix
            --> latent update
            --> prediction update
            --> ReverseEmbedding -> logits
    """

    def __init__(
        self,
        vocab_size,
        dim=128,
        hidden_dim=256,
        num_heads=4,
        num_experts=4,
        steps=4,
        causal=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.steps = steps

        # Input embedding
        self.embed = nn.Embedding(vocab_size, dim)
        self.rms = RMSNorm(dim)

        # Core blocks
        self.selector = SelectorGate(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, causal=causal)
        self.moe = SimpleMoE(dim, hidden_dim, num_experts=num_experts)

        # prediction state (latent -> pred space, same dim)
        self.pred_proj = nn.Linear(dim, dim)

        # Reverse embedding to vocab logits
        self.rev_embed = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, steps=None):
        """
        input_ids : [B,T] integer tokens
        returns:
            logits : [B,T,V]
        """
        if steps is None:
            steps = self.steps

        x = self.embed(input_ids)  # input embedding [B,T,D]
        latent = x
        prediction = torch.zeros_like(x)

        for _ in range(steps):
            # RMSNorm on (latent + input residual)
            u = self.rms(latent + x)

            # add prediction feedback
            u = u + prediction

            # selector gate
            g = self.selector(u)          # [B,T,1]
            attn_out = self.attn(u)      # [B,T,D]
            moe_out = self.moe(u)        # [B,T,D]

            # blend branches per token
            mix = g * attn_out + (1.0 - g) * moe_out

            # latent update (residual)
            latent = latent + mix

            # prediction update in latent space
            prediction = self.pred_proj(latent)

        # Reverse embedding to vocabulary logits
        logits = self.rev_embed(prediction)
        return logits


# ============================================
# Example usage / smoke test
# ============================================

if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 100
    model = TinyRecursiveMoEAttnModel(
        vocab_size=vocab_size,
        dim=64,
        hidden_dim=128,
        num_heads=4,
        num_experts=4,
        steps=3,
    )

    B, T = 2, 10
    x = torch.randint(0, vocab_size, (B, T))
    logits = model(x)
    print("logits shape:", logits.shape)  # [2,10,100]

    # dummy loss
    target = torch.randint(0, vocab_size, (B, T))
    loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    loss.backward()
    print("loss:", loss.item())
