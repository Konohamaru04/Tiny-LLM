from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.parent


def assert_exists(path: str | Path, description: str = "Path") -> Path:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {p}")
    return p


def read_json(path: str | Path) -> Any:
    p = assert_exists(path, "JSON file")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> None:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=p.parent,
        prefix=f".{p.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, p)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _pack_safetensors_state(
    value: Any,
    tensors: dict[str, torch.Tensor],
    tensor_references: dict[tuple[Any, ...], str],
) -> Any:
    if isinstance(value, torch.Tensor):
        identity = (
            str(value.device),
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.dtype),
        )
        existing_key = tensor_references.get(identity)
        if existing_key is not None:
            return {"__tensor__": existing_key}
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        tensor_references[identity] = key
        return {"__tensor__": key}
    if isinstance(value, dict):
        return {
            "__mapping__": [
                [
                    _pack_safetensors_state(key, tensors, tensor_references),
                    _pack_safetensors_state(item, tensors, tensor_references),
                ]
                for key, item in value.items()
            ]
        }
    if isinstance(value, tuple):
        return {
            "__tuple__": [
                _pack_safetensors_state(
                    item,
                    tensors,
                    tensor_references,
                )
                for item in value
            ]
        }
    if isinstance(value, list):
        return [
            _pack_safetensors_state(item, tensors, tensor_references)
            for item in value
        ]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "SafeTensors checkpoint metadata contains unsupported value type "
        f"{type(value).__name__}."
    )


def _unpack_safetensors_state(
    value: Any,
    tensors: dict[str, torch.Tensor],
) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__tensor__"}:
            key = str(value["__tensor__"])
            if key not in tensors:
                raise ValueError(f"SafeTensors checkpoint is missing tensor {key}.")
            return tensors[key]
        if set(value) == {"__tuple__"}:
            return tuple(
                _unpack_safetensors_state(item, tensors)
                for item in value["__tuple__"]
            )
        if set(value) == {"__mapping__"}:
            return {
                _unpack_safetensors_state(key, tensors):
                _unpack_safetensors_state(item, tensors)
                for key, item in value["__mapping__"]
            }
        return {
            key: _unpack_safetensors_state(item, tensors)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _unpack_safetensors_state(item, tensors) for item in value
        ]
    return value


def _save_safetensors_checkpoint(
    state: dict[str, Any],
    path: Path,
) -> Path:
    tensors: dict[str, torch.Tensor] = {}
    tensor_references: dict[tuple[Any, ...], str] = {}
    packed_state = _pack_safetensors_state(
        state,
        tensors,
        tensor_references,
    )
    metadata = {
        "format": "tiny-llm-training-state-v1",
        "state_json": json.dumps(
            packed_state,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        save_safetensors_file(
            tensors,
            str(temporary_path),
            metadata=metadata,
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def save_torch_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".safetensors":
        return _save_safetensors_checkpoint(state, p)
    torch.save(state, p)
    return p


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    p = assert_exists(path, "Checkpoint")
    if p.suffix.lower() == ".safetensors":
        device = str(map_location)
        try:
            tensors = load_safetensors_file(str(p), device=device)
            with safe_open(str(p), framework="pt", device=device) as handle:
                metadata = handle.metadata() or {}
            if metadata.get("format") != "tiny-llm-training-state-v1":
                raise ValueError(
                    "Unsupported or missing Tiny-LLM SafeTensors format metadata."
                )
            raw_state = metadata.get("state_json")
            if not raw_state:
                raise ValueError(
                    "SafeTensors checkpoint is missing state_json metadata."
                )
            state = _unpack_safetensors_state(
                json.loads(raw_state),
                tensors,
            )
            if not isinstance(state, dict):
                raise ValueError(
                    "SafeTensors checkpoint root must decode to a dictionary."
                )
            return state
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SafeTensors checkpoint from {p}: {exc}"
            ) from exc
    try:
        return torch.load(p, map_location=map_location, weights_only=True)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load checkpoint from {p}: {exc}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if device_str == "cpu":
        return torch.device("cpu")

    try:
        device = torch.device(device_str)
    except Exception as exc:
        raise ValueError(f"Invalid device string: {device_str}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device '{device_str}', but CUDA is not available.")
    return device


def get_amp_settings(device: torch.device, enabled: bool = True) -> dict[str, Any]:
    if not enabled or device.type != "cuda":
        return {"enabled": False, "dtype": None, "use_grad_scaler": False}

    if torch.cuda.is_bf16_supported():
        return {"enabled": True, "dtype": torch.bfloat16, "use_grad_scaler": False}

    return {"enabled": True, "dtype": torch.float16, "use_grad_scaler": True}


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return sum(p.numel() for p in params)


def human_count(value: int) -> str:
    units = ["", "K", "M", "B"]
    v = float(value)
    for unit in units:
        if abs(v) < 1000.0:
            return f"{v:.2f}{unit}"
        v /= 1000.0
    return f"{v:.2f}T"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "_orig_mod"):
        return model._orig_mod  # type: ignore[attr-defined]
    return model


def sha256_file(path: str | Path) -> str:
    p = assert_exists(path, "File")
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_checkpoint_fingerprints(
    state: dict[str, Any],
    model_config: Any | None = None,
    tokenizer_model_path: str | Path | None = None,
    context: str = "checkpoint",
) -> None:
    checkpoint_config_hash = state.get("model_config_hash")
    if checkpoint_config_hash and model_config is not None:
        expected = stable_json_hash(model_config)
        if checkpoint_config_hash != expected:
            raise ValueError(
                f"{context} model_config hash does not match the current runtime config.\n"
                f"checkpoint={checkpoint_config_hash}\n"
                f"runtime={expected}"
            )

    checkpoint_tokenizer_hash = state.get("tokenizer_sha256")
    if checkpoint_tokenizer_hash and tokenizer_model_path:
        runtime_tokenizer_hash = sha256_file(tokenizer_model_path)
        if checkpoint_tokenizer_hash != runtime_tokenizer_hash:
            raise ValueError(
                f"{context} tokenizer hash does not match the current tokenizer file.\n"
                f"checkpoint={checkpoint_tokenizer_hash}\n"
                f"runtime={runtime_tokenizer_hash}\n"
                f"tokenizer={resolve_path(tokenizer_model_path)}"
            )


def maybe_compile_model(
    model: torch.nn.Module,
    enabled: bool = False,
    backend: str = "",
    mode: str = "",
) -> torch.nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile was requested, but this PyTorch build does not expose torch.compile.")

    kwargs: dict[str, Any] = {}
    if backend:
        kwargs["backend"] = backend
    if mode:
        kwargs["mode"] = mode
    return torch.compile(model, **kwargs)
