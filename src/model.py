from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        input_dtype = x.dtype
        x_float = x.float()
        norm = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(norm + self.eps)
        return (self.weight.float() * normalized).to(dtype=input_dtype)


def build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == "layernorm":
        return nn.LayerNorm(config.n_embd, eps=config.norm_eps)
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.n_embd, eps=config.norm_eps)
    raise ValueError(f"Unsupported norm_type: {config.norm_type}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.kv_repeat = self.n_head // self.n_kv_head
        self.dropout = config.dropout
        self.attention_impl = config.attention_impl
        self.use_rope = config.positional_embedding == "rope"
        self.use_sdpa = self._select_attention_impl(config.attention_impl)

        self.qkv_proj = nn.Linear(
            config.n_embd,
            config.n_embd + (2 * self.kv_dim),
            bias=config.linear_bias,
        )
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.linear_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_rope:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires head_dim to be even.")
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        else:
            self.register_buffer("inv_freq", None, persistent=False)

        if not self.use_sdpa:
            mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)
        else:
            self.register_buffer("causal_mask", None, persistent=False)

    def _select_attention_impl(self, attention_impl: str) -> bool:
        has_sdpa = hasattr(F, "scaled_dot_product_attention")
        if attention_impl == "manual":
            return False
        if attention_impl == "sdpa":
            if not has_sdpa:
                raise RuntimeError("attention_impl='sdpa' was requested, but scaled_dot_product_attention is unavailable.")
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

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split((self.n_embd, self.kv_dim, self.kv_dim), dim=2)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self._apply_rope(self.q_norm(q))
        k = self._apply_rope(self.k_norm(k))

        if self.use_sdpa:
            try:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                    enable_gqa=self.n_head != self.n_kv_head,
                )
            except TypeError:  # PyTorch versions before native GQA support.
                k = k.repeat_interleave(self.kv_repeat, dim=1)
                v = v.repeat_interleave(self.kv_repeat, dim=1)
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
        else:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        raw_hidden_dim = config.n_embd * config.mlp_ratio
        hidden_dim = int(
            config.mlp_multiple_of
            * math.ceil(raw_hidden_dim / config.mlp_multiple_of)
        )
        self.mlp_type = config.mlp_type
        self.fc = nn.Linear(config.n_embd, hidden_dim, bias=config.linear_bias)
        self.gate = (
            nn.Linear(config.n_embd, hidden_dim, bias=config.linear_bias)
            if self.mlp_type == "swiglu"
            else None
        )
        self.proj = nn.Linear(hidden_dim, config.n_embd, bias=config.linear_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu":
            x = self.fc(x)
            x = F.gelu(x, approximate="tanh")
        elif self.mlp_type == "swiglu":
            if self.gate is None:
                raise RuntimeError("SwiGLU gate projection was not initialized.")
            x = F.silu(self.gate(x)) * self.fc(x)
        else:
            raise ValueError(f"Unsupported mlp_type: {self.mlp_type}")
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = build_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd) if config.positional_embedding == "learned" else None
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = build_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.gradient_checkpointing = bool(config.gradient_checkpointing)

        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        if config.residual_scale_init:
            residual_std = config.initializer_range / math.sqrt(2 * config.n_layer)
            for block in self.blocks:
                nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
                nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be rank 2 [batch, time], got shape {tuple(input_ids.shape)}")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )

        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must match input_ids shape {tuple(input_ids.shape)}"
            )

        positions = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long)
        tok = self.token_emb(input_ids)
        if self.pos_emb is not None:
            pos = self.pos_emb(positions)[None, :, :]
            x = self.drop(tok + pos)
        else:
            x = self.drop(tok)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss
