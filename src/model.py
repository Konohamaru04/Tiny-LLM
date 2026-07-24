from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import ModelConfig
from src.moe import SparseMoE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


def build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.attention_impl = config.attention_impl
        self.use_rope = config.positional_embedding == "rope"
        self.use_sdpa = self._select_attention_impl(config.attention_impl)

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires head_dim to be even.")
            inv_freq = 1.0 / (
                config.rope_theta
                ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("inv_freq", None, persistent=False)

        if not self.use_sdpa:
            mask = torch.tril(
                torch.ones(config.block_size, config.block_size, dtype=torch.bool)
            )
            self.register_buffer(
                "causal_mask",
                mask.view(1, 1, config.block_size, config.block_size),
                persistent=False,
            )
        else:
            self.register_buffer("causal_mask", None, persistent=False)

    def _select_attention_impl(self, attention_impl: str) -> bool:
        has_sdpa = hasattr(F, "scaled_dot_product_attention")
        if attention_impl == "manual":
            return False
        if attention_impl == "sdpa":
            if not has_sdpa:
                raise RuntimeError("scaled_dot_product_attention is unavailable")
            return True
        return has_sdpa

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_rope or self.inv_freq is None:
            return x
        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=x.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=x.dtype)
        return (x * cos) + (rotate_half(x) * sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        q, k, v = self.qkv_proj(x).split(self.n_embd, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(~mask, float("-inf"))
            att = self.attn_dropout(F.softmax(att, dim=-1))
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.resid_dropout(self.out_proj(y))


class DenseMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.n_embd * config.mlp_ratio
        self.mlp_type = config.mlp_type
        self.fc = nn.Linear(config.n_embd, hidden_dim)
        self.gate = (
            nn.Linear(config.n_embd, hidden_dim)
            if self.mlp_type == "swiglu"
            else None
        )
        self.proj = nn.Linear(hidden_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu":
            x = F.gelu(self.fc(x), approximate="tanh")
        elif self.mlp_type == "swiglu":
            if self.gate is None:
                raise RuntimeError("SwiGLU gate projection was not initialized")
            x = F.silu(self.gate(x)) * self.fc(x)
        else:
            raise ValueError(f"Unsupported mlp_type: {self.mlp_type}")
        return self.dropout(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int):
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config)

        num_experts = int(getattr(config, "moe_num_experts", 4))
        top_k = int(getattr(config, "moe_top_k", 2))
        every_n = int(getattr(config, "moe_every_n_layers", 1))
        router_jitter = float(getattr(config, "moe_router_jitter", 0.01))
        use_moe = num_experts > 1 and every_n > 0 and (layer_index + 1) % every_n == 0

        if use_moe:
            self.ffn: nn.Module = SparseMoE(
                dim=config.n_embd,
                hidden_dim=config.n_embd * config.mlp_ratio,
                num_experts=num_experts,
                top_k=top_k,
                dropout=config.dropout,
                router_jitter=router_jitter,
            )
        else:
            self.ffn = DenseMLP(config)

    @property
    def router_aux_loss(self) -> torch.Tensor | None:
        if isinstance(self.ffn, SparseMoE) and self.ffn.last_stats is not None:
            return self.ffn.last_stats.auxiliary_loss
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.positional_embedding == "learned"
            else None
        )
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, index) for index in range(config.n_layer)]
        )
        self.ln_f = build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)
        self.router_aux_loss_coef = float(
            getattr(config, "moe_aux_loss_coef", 0.01)
        )
        self.last_router_aux_loss: torch.Tensor | None = None

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _router_loss(self, device: torch.device) -> torch.Tensor:
        losses = [
            block.router_aux_loss
            for block in self.blocks
            if block.router_aux_loss is not None
        ]
        if not losses:
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be rank 2 [batch, time]")
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets shape must match input_ids shape")

        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        tok = self.token_emb(input_ids)
        x = self.drop(tok + self.pos_emb(positions)[None, :, :]) if self.pos_emb is not None else self.drop(tok)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        logits = self.lm_head(self.ln_f(x))
        router_loss = self._router_loss(input_ids.device)
        self.last_router_aux_loss = router_loss.detach()

        loss = None
        if targets is not None:
            language_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss = language_loss + self.router_aux_loss_coef * router_loss
        return logits, loss
