from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


_SAFE_CONTEXT_EXTENSION_FIELDS = {
    "attention_impl",
    "block_size",
    "gradient_checkpointing",
}
_SAFE_WEIGHT_ONLY_FIELDS = {
    "attention_impl",
    "dropout",
    "gradient_checkpointing",
    "moe_load_balance_loss_coefficient",
    "moe_router_jitter",
    "moe_router_z_loss_coefficient",
}


def _as_mapping(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass or mapping")


def context_extension_differences(
    checkpoint_config: Mapping[str, Any],
    runtime_config: Any,
) -> dict[str, tuple[Any, Any]]:
    checkpoint = dict(checkpoint_config)
    runtime = _as_mapping(runtime_config)
    keys = set(checkpoint) | set(runtime)
    return {
        key: (checkpoint.get(key), runtime.get(key))
        for key in sorted(keys)
        if checkpoint.get(key) != runtime.get(key)
    }


def is_safe_context_extension(
    checkpoint_config: Mapping[str, Any],
    runtime_config: Any,
) -> bool:
    checkpoint = dict(checkpoint_config)
    runtime = _as_mapping(runtime_config)
    if checkpoint.get("positional_embedding") != "rope":
        return False
    if runtime.get("positional_embedding") != "rope":
        return False
    if runtime.get("attention_impl") not in {"auto", "sdpa"}:
        return False
    if checkpoint.get("attention_impl") not in {"auto", "sdpa"}:
        return False

    old_block_size = checkpoint.get("block_size")
    new_block_size = runtime.get("block_size")
    if not isinstance(old_block_size, int) or not isinstance(new_block_size, int):
        return False
    if new_block_size <= old_block_size:
        return False

    differences = context_extension_differences(checkpoint, runtime)
    return bool(differences) and set(differences).issubset(
        _SAFE_CONTEXT_EXTENSION_FIELDS
    )


def is_weight_compatible(
    checkpoint_config: Mapping[str, Any],
    runtime_config: Any,
) -> bool:
    """Return true when configs differ only in parameter-free behavior."""
    differences = context_extension_differences(
        checkpoint_config,
        runtime_config,
    )
    return set(differences).issubset(_SAFE_WEIGHT_ONLY_FIELDS)
