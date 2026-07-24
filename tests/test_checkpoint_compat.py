from __future__ import annotations

from dataclasses import asdict, replace

from src.checkpoint_compat import (
    context_extension_differences,
    is_safe_context_extension,
)
from src.config import ModelConfig


def _moe_rope_config(block_size: int = 4096) -> ModelConfig:
    return ModelConfig(
        vocab_size=4096,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        mlp_ratio=2,
        dropout=0.0,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        positional_embedding="rope",
        rope_theta=1_000_000.0,
        attention_impl="sdpa",
        gradient_checkpointing=True,
        moe_num_experts=4,
        moe_top_k=2,
        moe_every_n_layers=1,
        moe_aux_loss_coef=0.01,
        moe_router_jitter=0.01,
    )


def test_larger_rope_context_is_safe_model_only_warm_start() -> None:
    source = _moe_rope_config(4096)
    target = replace(source, block_size=16384)
    assert is_safe_context_extension(asdict(source), target)
    assert context_extension_differences(asdict(source), target) == {
        "block_size": (4096, 16384)
    }


def test_expert_count_change_is_not_context_compatible() -> None:
    source = _moe_rope_config()
    target = replace(source, block_size=16384, moe_num_experts=8)
    assert not is_safe_context_extension(asdict(source), target)


def test_context_shrink_is_not_safe() -> None:
    source = _moe_rope_config(16384)
    target = replace(source, block_size=4096)
    assert not is_safe_context_extension(asdict(source), target)


def test_learned_positions_cannot_use_context_warm_start() -> None:
    source = replace(_moe_rope_config(), positional_embedding="learned")
    target = replace(source, block_size=16384)
    assert not is_safe_context_extension(asdict(source), target)
