import torch
import torch.nn as nn
import torch.nn.functional as F

class CostAwareHeteroMoE(nn.Module):
    def __init__(
        self,
        dim,
        experts_dim,            # list: [[64,32],[256,128,64],32,...] in latent space
        groups=1,
        top_k=2,
        latent_dim=0.5,
        routing="topk",
        shared_expert=True,
        use_core=True,
        use_erc_loss=False,
        cost_lambda=0.0005      # cost penalty strength
    ):
        super().__init__()

        self.dim = dim
        self.latent = int(dim * latent_dim)
        self.experts_dim = experts_dim
        self.num_experts = len(experts_dim)
        self.groups = groups
        self.top_k = top_k
        self.routing = routing
        self.use_core = use_core
        self.use_erc_loss = use_erc_loss
        self.cost_lambda = cost_lambda

        # ↓ latent compression
        self.down = nn.Linear(dim, self.latent)
        self.up   = nn.Linear(self.latent, dim)

        self.experts = nn.ModuleList()
        cost_list = []

        for config in experts_dim:
            layers = []
            in_d = self.latent

            if isinstance(config, int):
                config = [config]

            cost = 0
            for h in config:
                layers.append(nn.GELU())
                layers.append(nn.Linear(in_d, h))
                cost += in_d * h
                in_d = h

            layers.append(nn.GELU())
            layers.append(nn.Linear(in_d, self.latent))
            cost += in_d * self.latent

            self.experts.append(nn.Sequential(*layers))
            cost_list.append(cost)

        # register cost as buffer so it moves with .to(device)
        self.register_buffer("expert_cost", torch.tensor(cost_list, dtype=torch.float))

        # Shared expert
        self.shared = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.latent, self.latent),
            nn.GELU(),
            nn.Linear(self.latent, self.latent)
        ) if shared_expert else None

        # Router head (uses *original* dim)
        self.router = nn.Linear(dim, self.num_experts)

        # Residual Core (RMoE)
        self.core = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim)
        ) if use_core else None

    def route(self, x):
        logits = self.router(x)                  # [B,T,E]
        cost = self.expert_cost                  # [E]

        # Penalize expensive experts
        logits = logits - self.cost_lambda * cost  # <<< cost-awareness

        probs = F.softmax(logits, -1)

        topv, top_i = probs.topk(self.top_k, dim=-1)
        gate = F.softmax(topv, dim=-1).unsqueeze(-1)  # [B,T,k,1]
        return top_i, gate, probs

    def forward(self, x, return_erc=False):
        """
        x : [B,T,D]
        returns:
          out or (out, erc_loss)
        """
        B, T, _ = x.size()
        h = self.down(x)           # [B,T,latent]

        idx, gate, p_all = self.route(x)
        out = torch.zeros_like(h)

        for k in range(self.top_k):
            e_idx = idx[..., k]                           # [B,T]
            mask = F.one_hot(e_idx, self.num_experts).float()  # [B,T,E]

            batch_out = torch.zeros_like(h)
            for e_id, expert in enumerate(self.experts):
                m = mask[..., e_id].unsqueeze(-1)         # [B,T,1]
                if m.sum() > 0:
                    batch_out = batch_out + expert(h * m)

            out = out + batch_out * gate[..., k, :]

        if self.shared is not None:
            out = out + 0.1 * self.shared(h)

        out = self.up(out)                                # back to [B,T,D]

        if self.core is not None:
            out = out + self.core(x)

        if self.use_erc_loss and return_erc:
            # ERC loss: decorrelate router rows
            W = self.router.weight                        # [E,D] or [num_experts,D]
            M = (W @ W.T).abs()                           # [E,E]
            diag = M.diag().unsqueeze(1)
            erc = ((M - 0.7 * diag).clamp(min=0)).mean()
            return out, erc

        return out

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
        k = int(N * self.clip_ratio) if self.topk is None else self.topk
        k = max(k, 1)
        thr = torch.topk(scores, k, dim=-1).values[..., -1].unsqueeze(-1)
        return scores.masked_fill(scores < thr, float('-inf'))

    @torch.no_grad()
    def update_cache(self, K, V):
        K = K.detach(); V = V.detach()
        if self.K_mem is None:
            self.K_mem = K
            self.V_mem = V
        else:
            self.K_mem = torch.cat([self.K_mem, K], dim=1)
            self.V_mem = torch.cat([self.V_mem, V], dim=1)
            if self.K_mem.size(1) > self.num_kv:
                self.K_mem = self.K_mem[:, -self.num_kv:]
                self.V_mem = self.V_mem[:, -self.num_kv:]

    def forward(self, x, mode="train", cache_update=False, use_sparse=True):
        # x: [B,T,D]
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        if mode == "train":
            att = (Q @ K.transpose(-1, -2)) * self.scale
            att = att.softmax(-1)
            return self.o(att @ V)

        if cache_update:
            self.update_cache(K, V)

        Km, Vm = self.K_mem, self.V_mem
        α = torch.sigmoid(self.gate)
        τ = self.log_tau.exp()

        scores = (α * self.beta_q + (1 - α) * self.beta_k) * (Q @ Km.transpose(-1, -2)) * self.scale * τ
        scores -= scores.max(-1, keepdim=True).values

        if use_sparse:
            scores = self.qk_clip(scores)

        att = scores.softmax(-1)
        return self.o(att @ Vm)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., D)
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

class SelectorBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_kv=8,
        moe_experts_dim=None,
        moe_latent_dim=0.5,
        moe_top_k=2,
        moe_routing="topk",
        moe_shared=True,
        moe_use_core=True,
        moe_use_erc_loss=False,
        moe_cost_lambda=5e-4,
    ):
        super().__init__()
        self.dim = dim

        self.norm = RMSNorm(dim)

        # MKV Attention
        self.attn = FusedSparseMKVAttention(dim=dim, num_kv=num_kv)

        # Cost-aware MoE
        self.moe = CostAwareHeteroMoE(
            dim=dim,
            experts_dim=moe_experts_dim,
            top_k=moe_top_k,
            latent_dim=moe_latent_dim,
            routing=moe_routing,
            shared_expert=moe_shared,
            use_core=moe_use_core,
            use_erc_loss=moe_use_erc_loss,
            cost_lambda=moe_cost_lambda,
        )

    def forward(self, x, mode="train", need_reg=False):
        """
        return: out + optional reg
        """
        h = self.norm(x)

        # --- Compute branches ---
        attn_out = self.attn(h, mode=mode)              # [B,T,D]

        if need_reg and self.moe.use_erc_loss:
            moe_out, erc_loss = self.moe(h, return_erc=True)
        else:
            moe_out = self.moe(h)
            erc_loss = torch.tensor(0., device=x.device)
        # ====== Fused aggregation (residual) ======
        # Simple fusion: equal contribution. Later can make weighted or adaptive.
        out = x + attn_out + moe_out


        # Optional loss return (MoE ERC + Path regularizer)
        if need_reg:
            reg =  erc_loss
            return out, reg
        return out

class DeepImprovementTRM(nn.Module):
    """
    Deep-Improvement TRM for FEATURE INPUTS (not token IDs)

    - x: [B, N, F] float features (N=sequence length, can be 1)
    - internal dim = model width
    - states:
        X : context / input-transformed
        Y : prediction state
        Z : latent reasoning state
    """

    def __init__(
        self,
        vocab_size: int,          # number of output classes (3 for DOWN/NEUTRAL/UP)
        dim: int,
        input_dim: int,           # FEATURE dimension from PatternDataset
        num_kv: int = 8,
        num_outer_steps: int = 3,
        num_latent_steps: int = 4,
        moe_experts_dim=None,
        moe_latent_dim: float = 0.5,
        moe_top_k: int = 2,
        moe_routing: str = "topk",
        moe_shared: bool = True,
        moe_use_core: bool = True,
        moe_use_erc_loss: bool = False,
        moe_cost_lambda: float = 5e-4,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_outer_steps = num_outer_steps
        self.num_latent_steps = num_latent_steps

        # FEATURE → model-dim projection (replaces Embedding)
        self.input_proj = nn.Linear(input_dim, dim)

        # learned initial Y, Z (per position)
        self.y_init = nn.Parameter(torch.randn(dim) * 1e-2)
        self.z_init = nn.Parameter(torch.randn(dim) * 1e-2)

        # shared refinement block (Attn + Cost-Aware MoE)
        self.block = SelectorBlock(
            dim=dim,
            num_kv=num_kv,
            moe_experts_dim=moe_experts_dim,
            moe_latent_dim=moe_latent_dim,
            moe_top_k=moe_top_k,
            moe_routing=moe_routing,
            moe_shared=moe_shared,
            moe_use_core=moe_use_core,
            moe_use_erc_loss=moe_use_erc_loss,
            moe_cost_lambda=moe_cost_lambda,
        )

        # prediction head from Y
        self.output_head = nn.Linear(dim, vocab_size)
        
    def iter_qk_pairs(self):
     # scan all submodules for attention layers containing q/k projections
     for name, layer in self.named_modules():

        # Standard q/k projection names
        for q_name, k_name in [
            ("q_proj", "k_proj"),
            ("q", "k"),
            ("qA", "kA"),           # Muon latent attn phase A
            ("qB", "kB"),           # Muon latent attn phase B
            ("q_latent", "k_latent"),
            ("q_outer", "k_outer"),
            ("qkv_q", "qkv_k"),
        ]:
            if hasattr(layer, q_name) and hasattr(layer, k_name):
                q = getattr(layer, q_name)
                k = getattr(layer, k_name)

                # must be Linear & have head count attribute else skip
                heads = getattr(layer, "num_heads", getattr(self, "num_heads", 1))
                
                if isinstance(q, torch.nn.Linear) and isinstance(k, torch.nn.Linear):
                    yield q, k, heads


    def _init_states(self, x_emb: torch.Tensor):
        """
        x_emb: [B, N, D]
        returns X, Y, Z all [B, N, D]
        """
        B, N, D = x_emb.shape
        X = x_emb

        y = self.y_init.view(1, 1, D).expand(B, N, D)
        z = self.z_init.view(1, 1, D).expand(B, N, D)
        return X, y, z

    def _apply_block_on_concat(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Z: torch.Tensor,
        update: str = "z",
        mode: str = "train",
        need_reg: bool = False,
    ):
        """
        X, Y, Z: [B, N, D]
        concat along sequence dim: [B, 3N, D]
        run SelectorBlock → split back
        update: "z" → update Z only, "y" → update Y only, "both" → update Y & Z
        """
        B, N, D = X.shape
        concat = torch.cat([X, Y, Z], dim=1)  # [B, 3N, D]

        if need_reg:
            out, reg = self.block(concat, mode=mode, need_reg=True)
        else:
            out = self.block(concat, mode=mode, need_reg=False)
            reg = torch.tensor(0.0, device=concat.device)

        X_o = out[:, :N, :]
        Y_o = out[:, N:2 * N, :]
        Z_o = out[:, 2 * N:, :]

        if update == "z":
            return X, Y, Z_o, reg
        elif update == "y":
            return X, Y_o, Z, reg
        else:  # "both"
            return X, Y_o, Z_o, reg

    def latent_reasoning(self, X, Y, Z, mode="train"):
        """
        Latent loop:
        - repeat num_latent_steps times: update Z
        - once: update Y
        Accumulate regularization from MoE / ERC.
        """
        total_reg = 0.0

        # refine Z several times
        for _ in range(self.num_latent_steps):
            X, Y, Z, reg = self._apply_block_on_concat(
                X, Y, Z, update="z", mode=mode, need_reg=True
            )
            total_reg = total_reg + reg

        # then refine Y once
        X, Y, Z, reg = self._apply_block_on_concat(
            X, Y, Z, update="y", mode=mode, need_reg=True
        )
        total_reg = total_reg + reg

        return X, Y, Z, total_reg
      
    def forward(self, x, mode: str = "train", need_reg: bool = False):
        """
        x: [B, F] or [B, N, F] float features
        returns:
          logits: [B, N, V]
          optionally with reg term
        """
        if x.dim() == 2:
            # treat as single-token sequence
            x = x.unsqueeze(1)  # [B, 1, F]

        # project features → model dim
        x_emb = self.input_proj(x)  # [B, N, D]
        X, Y, Z = self._init_states(x_emb)

        total_reg = 0.0
        for _ in range(self.num_outer_steps):
            X, Y, Z, reg = self.latent_reasoning(X, Y, Z, mode=mode)
            total_reg = total_reg + reg

        logits = self.output_head(Y)  # [B, N, V]

        if need_reg:
            return logits, total_reg
        return logits
