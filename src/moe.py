from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEStats:
    load_balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    expert_usage: torch.Tensor
    mean_router_probability: torch.Tensor


class ExpertSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        dropout: float,
        linear_bias: bool,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=linear_bias)
        self.up = nn.Linear(dim, hidden_dim, bias=linear_bias)
        self.down = nn.Linear(hidden_dim, dim, bias=linear_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class SparseMoE(nn.Module):
    """Capacity-free token-choice sparse MoE with normalized top-k routing."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        num_experts: int,
        top_k: int,
        dropout: float,
        linear_bias: bool,
        router_jitter: float,
        shared_expert: bool,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2 for sparse MoE.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts].")
        if router_jitter < 0.0 or router_jitter >= 1.0:
            raise ValueError("router_jitter must be in [0, 1).")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_jitter = float(router_jitter)
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            ExpertSwiGLU(
                dim,
                hidden_dim,
                dropout=dropout,
                linear_bias=linear_bias,
            )
            for _ in range(num_experts)
        )
        self.shared_expert = (
            ExpertSwiGLU(
                dim,
                hidden_dim,
                dropout=dropout,
                linear_bias=linear_bias,
            )
            if shared_expert
            else None
        )
        self.last_stats: MoEStats | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_input = flat
        if self.training and self.router_jitter:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.router_jitter,
                1.0 + self.router_jitter,
            )
            router_input = router_input * noise

        # Keep routing decisions in fp32 even under autocast.
        router_logits = F.linear(
            router_input.float(),
            self.router.weight.float(),
        )
        router_probabilities = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(
            router_probabilities,
            self.top_k,
            dim=-1,
        )
        top_weights = top_weights / top_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-9)

        token_count = flat.size(0)
        token_indices = torch.arange(
            token_count,
            device=flat.device,
        ).repeat_interleave(self.top_k)
        assignment_experts = top_indices.reshape(-1)
        assignment_weights = top_weights.reshape(-1).to(flat.dtype)

        # Sort once, then each expert receives a contiguous token batch.
        order = torch.argsort(assignment_experts)
        sorted_experts = assignment_experts[order]
        sorted_tokens = token_indices[order]
        sorted_weights = assignment_weights[order]
        expert_counts = torch.bincount(
            sorted_experts,
            minlength=self.num_experts,
        )

        # One device synchronization obtains all split sizes, rather than one
        # synchronization plus a full token scan per expert.
        split_sizes = expert_counts.tolist()
        token_chunks = torch.split(sorted_tokens, split_sizes)
        weight_chunks = torch.split(sorted_weights, split_sizes)
        expert_outputs = [
            expert(flat[token_chunk]) * weight_chunk.unsqueeze(-1)
            for expert, token_chunk, weight_chunk in zip(
                self.experts,
                token_chunks,
                weight_chunks,
            )
            if token_chunk.numel()
        ]
        weighted_outputs = torch.cat(expert_outputs, dim=0)
        output = torch.zeros_like(flat)
        output.index_add_(0, sorted_tokens, weighted_outputs)
        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        assignment_fraction = expert_counts.float() / float(
            max(1, token_count * self.top_k)
        )
        mean_probability = router_probabilities.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(
            assignment_fraction * mean_probability
        )
        router_z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
        self.last_stats = MoEStats(
            load_balance_loss=load_balance_loss,
            router_z_loss=router_z_loss,
            expert_usage=expert_counts.detach(),
            mean_router_probability=mean_probability.detach(),
        )
        return output.reshape(original_shape)
