from __future__ import annotations

import torch

from src.moe import SparseMoE


def test_moe_output_shape_and_usage() -> None:
    layer = SparseMoE(dim=16, hidden_dim=32, num_experts=4, top_k=2)
    inputs = torch.randn(2, 5, 16)
    outputs = layer(inputs)
    assert outputs.shape == inputs.shape
    assert layer.last_stats is not None
    assert layer.last_stats.expert_usage.shape == (4,)
    assert int(layer.last_stats.expert_usage.sum().item()) == 20
    assert torch.isfinite(layer.last_stats.auxiliary_loss)


def test_moe_supports_gradient_updates() -> None:
    layer = SparseMoE(dim=8, hidden_dim=16, num_experts=4, top_k=2)
    inputs = torch.randn(3, 4, 8, requires_grad=True)
    outputs = layer(inputs)
    assert layer.last_stats is not None
    loss = outputs.square().mean() + 0.01 * layer.last_stats.auxiliary_loss
    loss.backward()
    assert layer.router.weight.grad is not None
    assert any(module.down.weight.grad is not None for module in layer.experts)
