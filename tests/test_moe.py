from __future__ import annotations

import unittest

import torch

from src.moe import SparseMoE


class SparseMoETests(unittest.TestCase):
    def test_capacity_free_top_k_routes_every_assignment(self) -> None:
        layer = SparseMoE(
            dim=16,
            hidden_dim=32,
            num_experts=4,
            top_k=2,
            dropout=0.0,
            linear_bias=False,
            router_jitter=0.0,
            shared_expert=True,
        )
        inputs = torch.randn(2, 7, 16, requires_grad=True)

        outputs = layer(inputs)
        outputs.square().mean().backward()

        self.assertEqual(outputs.shape, inputs.shape)
        self.assertIsNotNone(layer.last_stats)
        assert layer.last_stats is not None
        self.assertEqual(int(layer.last_stats.expert_usage.sum()), 2 * 7 * 2)
        self.assertTrue(torch.isfinite(layer.last_stats.load_balance_loss))
        self.assertTrue(torch.isfinite(layer.last_stats.router_z_loss))
        self.assertIsNotNone(layer.router.weight.grad)

    def test_invalid_router_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SparseMoE(
                dim=8,
                hidden_dim=16,
                num_experts=4,
                top_k=5,
                dropout=0.0,
                linear_bias=False,
                router_jitter=0.0,
                shared_expert=True,
            )
