from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


_CONTEXT_EXTENSION_FIELDS = {"block_size", "gradient_checkpointing"}


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

    old_block = checkpoint.get("block_size")
    new_block = runtime.get("block_size")
    if not isinstance(old_block, int) or not isinstance(new_block, int):
        return False
    if new_block < old_block:
        return False

    differences = context_extension_differences(checkpoint, runtime)
    return bool(differences) and set(differences).issubset(_CONTEXT_EXTENSION_FIELDS)
