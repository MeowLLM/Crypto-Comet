import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedSparseMKVAttention(nn.Module):
    def __init__(self, dim, num_kv=8, clip_ratio=0.85, topk=None):
        super().__init__()
        self.dim = dim
        self.num_kv = num_kv
        self.scale = dim ** -0.5

        # fused projection (single GEMM = faster)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.o   = nn.Linear(dim, dim, bias=False)

        # fusion/gating
        self.beta_q = nn.Parameter(torch.tensor(1.0))
        self.beta_k = nn.Parameter(torch.tensor(1.0))
        self.gate   = nn.Parameter(torch.tensor(0.0))
        self.log_tau= nn.Parameter(torch.tensor(0.0))

        # streaming cache
        self.register_buffer("K_mem", None)
        self.register_buffer("V_mem", None)

        self.clip_ratio = clip_ratio
        self.topk = topk

    @torch.no_grad()
    def qk_clip(self, scores):
        N = scores.size(-1)
        k = int(N*self.clip_ratio) if self.topk is None else self.topk
        k = max(k, 1)
        thr = torch.topk(scores, k, dim=-1).values[..., -1].unsqueeze(-1)
        return scores.masked_fill(scores < thr, float('-inf'))

    @torch.no_grad()
    def update_cache(self, K, V):
        K = K.detach(); V = V.detach()
        if self.K_mem is None:
            self.K_mem = K; self.V_mem = V
        else:
            self.K_mem = torch.cat([self.K_mem, K], dim=1)
            self.V_mem = torch.cat([self.V_mem, V], dim=1)
            if self.K_mem.size(1) > self.num_kv:
                self.K_mem = self.K_mem[:, -self.num_kv:]
                self.V_mem = self.V_mem[:, -self.num_kv:]

    def forward(self, x, mode="train", cache_update=False, use_sparse=True):
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        if mode == "train":
            att = (Q @ K.transpose(-1,-2)) * self.scale
            att = att.softmax(-1)
            return self.o(att @ V)

        if cache_update: self.update_cache(K, V)

        Km, Vm = self.K_mem, self.V_mem
        α = torch.sigmoid(self.gate)
        τ = self.log_tau.exp()

        scores = (α*self.beta_q + (1-α)*self.beta_k)*(Q @ Km.transpose(-1,-2))*self.scale*τ
        scores -= scores.max(-1, keepdim=True).values

        if use_sparse:
            scores = self.qk_clip(scores)

        att = scores.softmax(-1)
        return self.o(att @ Vm)
