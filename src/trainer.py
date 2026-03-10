from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.utils import (
    count_parameters,
    ensure_dir,
    get_amp_settings,
    human_count,
    load_torch_checkpoint,
    save_torch_checkpoint,
)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "norm" not in name.lower() and "embedding" not in name.lower():
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    min_lr_ratio = min_lr / base_lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def infinite_loader(loader: DataLoader) -> Iterable:
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        model_config: ModelConfig,
        training_config: Any,
        tokenizer_model_path: str | Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        task_name: str,
    ):
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer_model_path = str(tokenizer_model_path)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.task_name = task_name

        self.model.to(self.device)

        self.optimizer = build_optimizer(
            self.model,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            warmup_steps=self.training_config.warmup_steps,
            max_steps=self.training_config.max_steps,
            min_lr=self.training_config.min_lr,
            base_lr=self.training_config.learning_rate,
        )

        amp = get_amp_settings(self.device, enabled=self.training_config.use_amp)
        self.amp_enabled = amp["enabled"]
        self.amp_dtype = amp["dtype"]
        self.use_grad_scaler = amp["use_grad_scaler"]
        self.scaler = torch.cuda.amp.GradScaler('cuda', enabled=self.use_grad_scaler)

        self.output_dir = ensure_dir(self.training_config.output_dir)
        self.latest_path = self.output_dir / "latest.pt"
        self.best_path = self.output_dir / "best.pt"

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.bad_eval_count = 0
        self.last_val_loss = float("inf")

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "bad_eval_count": self.bad_eval_count,
            "last_val_loss": self.last_val_loss,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "tokenizer_model_path": self.tokenizer_model_path,
        }

    def save_checkpoint(self, path: str | Path, label: str) -> Path:
        checkpoint_path = save_torch_checkpoint(self._checkpoint_state(), path)
        print(f"[checkpoint] saved {label}: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        state = load_torch_checkpoint(path, map_location="cpu")

        required = ["model_state", "optimizer_state", "scheduler_state", "global_step", "model_config"]
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(
                f"Checkpoint is missing required fields {missing}: {Path(path)}"
            )

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])

        scaler_state = state.get("scaler_state")
        if scaler_state is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)

        self.global_step = int(state.get("global_step", 0))
        self.best_val_loss = float(state.get("best_val_loss", float("inf")))
        self.bad_eval_count = int(state.get("bad_eval_count", 0))
        self.last_val_loss = float(state.get("last_val_loss", float("inf")))

        print(f"[resume] loaded checkpoint: {Path(path)}")
        print(f"[resume] global_step={self.global_step}, best_val_loss={self.best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []

        with self._autocast_context():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= self.training_config.eval_steps:
                    break
                x, y = batch
                x = x.to(self.device, non_blocking=self.device.type == "cuda")
                y = y.to(self.device, non_blocking=self.device.type == "cuda")
                _, loss = self.model(x, y)
                if loss is None:
                    raise RuntimeError("Model returned no loss during evaluation.")
                losses.append(float(loss.detach().cpu().item()))

        self.model.train()

        if not losses:
            raise RuntimeError("Validation loader produced zero batches during evaluation.")

        return sum(losses) / len(losses)

    def train(self) -> None:
        print(f"[train] task={self.task_name}")
        print(f"[train] device={self.device}")
        print(f"[train] parameters={human_count(count_parameters(self.model))}")
        print(
            f"[train] amp_enabled={self.amp_enabled}, "
            f"amp_dtype={self.amp_dtype}, grad_scaler={self.scaler.is_enabled()}"
        )

        train_iter = infinite_loader(self.train_loader)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        log_steps = 0
        tokens_since_log = 0
        log_start_time = time.time()

        while self.global_step < self.training_config.max_steps:
            step_loss = 0.0

            for _ in range(self.training_config.gradient_accumulation_steps):
                x, y = next(train_iter)
                x = x.to(self.device, non_blocking=self.device.type == "cuda")
                y = y.to(self.device, non_blocking=self.device.type == "cuda")

                with self._autocast_context():
                    _, loss = self.model(x, y)
                    if loss is None:
                        raise RuntimeError("Model returned no loss during training.")
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite loss encountered at global_step={self.global_step}: {loss.item()}"
                        )
                    loss = loss / self.training_config.gradient_accumulation_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_loss += float(loss.detach().cpu().item())
                tokens_since_log += int(x.numel())

            if self.training_config.grad_clip > 0.0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

            self.global_step += 1
            running_loss += step_loss
            log_steps += 1

            if self.global_step == 1 or self.global_step % self.training_config.log_interval == 0:
                elapsed = max(time.time() - log_start_time, 1e-6)
                avg_loss = running_loss / max(1, log_steps)
                lr = self.optimizer.param_groups[0]["lr"]
                toks_per_sec = tokens_since_log / elapsed

                print(
                    f"[train] step={self.global_step:06d} "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr:.6e} "
                    f"tok/s={toks_per_sec:,.0f}"
                )

                running_loss = 0.0
                log_steps = 0
                tokens_since_log = 0
                log_start_time = time.time()

            should_eval = (
                self.global_step % self.training_config.eval_interval == 0
                or self.global_step == self.training_config.max_steps
            )
            should_save = (
                self.global_step % self.training_config.save_interval == 0
                or self.global_step == self.training_config.max_steps
            )

            if should_eval:
                self.last_val_loss = self.evaluate()
                print(f"[eval] step={self.global_step:06d} val_loss={self.last_val_loss:.4f}")

                if self.last_val_loss < self.best_val_loss - 1e-6:
                    self.best_val_loss = self.last_val_loss
                    self.bad_eval_count = 0
                    self.save_checkpoint(self.best_path, "best")
                else:
                    self.bad_eval_count += 1
                    print(
                        f"[eval] no improvement. "
                        f"bad_eval_count={self.bad_eval_count}/{self.training_config.patience}"
                    )

                if self.training_config.patience > 0 and self.bad_eval_count >= self.training_config.patience:
                    self.save_checkpoint(self.latest_path, "latest")
                    snapshot = self.output_dir / f"step_{self.global_step:07d}.pt"
                    self.save_checkpoint(snapshot, f"snapshot@{self.global_step}")
                    print("[early-stop] patience exhausted. Stopping training.")
                    return

            if should_save:
                self.save_checkpoint(self.latest_path, "latest")
                snapshot = self.output_dir / f"step_{self.global_step:07d}.pt"
                self.save_checkpoint(snapshot, f"snapshot@{self.global_step}")

        print("[train] finished max_steps.")