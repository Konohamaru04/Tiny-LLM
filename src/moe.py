from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEStats:
    auxiliary_loss: torch.Tensor
    router_probabilities: torch.Tensor
    expert_usage: torch.Tensor


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SparseMoE(nn.Module):
    """Token-level top-k sparse Mixture-of-Experts layer.

    The implementation prioritizes clarity and correctness for small local
    models. Only selected experts process each token. A Switch-style auxiliary
    loss encourages balanced routing and is exposed to the training loop.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.0,
        router_jitter: float = 0.0,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            ExpertMLP(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        )
        self.last_stats: MoEStats | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])

        router_input = flat
        if self.training and self.router_jitter > 0.0:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.router_jitter,
                1.0 + self.router_jitter,
            )
            router_input = router_input * noise

        router_logits = self.router(router_input)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(flat.dtype)
        top_weights, top_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        dispatch = F.one_hot(top_indices, num_classes=self.num_experts).to(flat.dtype)

        for expert_index, expert in enumerate(self.experts):
            token_positions, slot_positions = torch.where(top_indices == expert_index)
            if token_positions.numel() == 0:
                continue
            expert_output = expert(flat[token_positions])
            weights = top_weights[token_positions, slot_positions].unsqueeze(-1)
            output.index_add_(0, token_positions, expert_output * weights)

        importance = router_probs.mean(dim=0)
        load = dispatch.sum(dim=1).mean(dim=0) / float(self.top_k)
        auxiliary_loss = self.num_experts * torch.sum(importance * load)
        expert_usage = dispatch.sum(dim=(0, 1))
        self.last_stats = MoEStats(
            auxiliary_loss=auxiliary_loss,
            router_probabilities=importance.detach(),
            expert_usage=expert_usage.detach(),
        )
        return output.reshape(original_shape)
