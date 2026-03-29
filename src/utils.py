from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_torch_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, p)
    return p


def load_torch_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    p = assert_exists(path, "Checkpoint")
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