from __future__ import annotations

import unittest
from dataclasses import asdict, replace

from src.checkpoint_compat import (
    context_extension_differences,
    is_safe_context_extension,
    is_weight_compatible,
)
from src.config import ModelConfig


class CheckpointCompatibilityTests(unittest.TestCase):
    def _rope_config(self, block_size: int = 2048) -> ModelConfig:
        return ModelConfig(
            vocab_size=8192,
            block_size=block_size,
            n_layer=4,
            n_head=4,
            n_kv_head=2,
            n_embd=128,
            mlp_ratio=2,
            dropout=0.0,
            positional_embedding="rope",
            gradient_checkpointing=True,
        )

    def test_larger_rope_context_is_safe_model_only_warm_start(self) -> None:
        source = self._rope_config(2048)
        target = replace(source, block_size=4096, attention_impl="sdpa")

        self.assertTrue(is_safe_context_extension(asdict(source), target))
        self.assertEqual(
            context_extension_differences(asdict(source), target),
            {
                "attention_impl": ("auto", "sdpa"),
                "block_size": (2048, 4096),
            },
        )

    def test_architecture_change_is_not_context_compatible(self) -> None:
        source = self._rope_config()
        target = replace(source, block_size=4096, n_kv_head=1)

        self.assertFalse(is_safe_context_extension(asdict(source), target))

    def test_context_shrink_and_learned_positions_are_rejected(self) -> None:
        source = self._rope_config(4096)
        self.assertFalse(
            is_safe_context_extension(
                asdict(source),
                replace(source, block_size=2048),
            )
        )
        learned = replace(source, positional_embedding="learned")
        self.assertFalse(
            is_safe_context_extension(
                asdict(learned),
                replace(learned, block_size=8192),
            )
        )

    def test_sft_can_disable_router_jitter_without_changing_weights(self) -> None:
        source = self._rope_config()
        target = replace(source, moe_router_jitter=0.0)

        self.assertTrue(is_weight_compatible(asdict(source), target))
        self.assertFalse(
            is_weight_compatible(
                asdict(source),
                replace(target, moe_num_experts=8),
            )
        )
