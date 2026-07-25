from __future__ import annotations

import csv
import inspect
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.evaluation import load_eval_prompts, run_generation_eval
from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import (
    count_parameters,
    ensure_dir,
    ensure_parent_dir,
    get_amp_settings,
    human_count,
    load_torch_checkpoint,
    sha256_file,
    resolve_path,
    save_torch_checkpoint,
    stable_json_hash,
    unwrap_model,
    verify_checkpoint_fingerprints,
    write_json,
)


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    fused: bool = True,
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

    kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    all_cuda = bool(decay_params or no_decay_params) and all(
        param.device.type == "cuda" for param in decay_params + no_decay_params
    )
    if supports_fused:
        kwargs["fused"] = bool(fused and all_cuda)
    return torch.optim.AdamW(optim_groups, **kwargs)


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
            fused=bool(getattr(self.training_config, "fused_optimizer", True)),
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
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_grad_scaler)
        else:  # pragma: no cover
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

        self.output_dir = ensure_dir(self.training_config.output_dir)
        self.latest_path = self.output_dir / "latest.safetensors"
        self.best_path = self.output_dir / "best.safetensors"
        self.metrics_jsonl_path = (
            resolve_path(self.training_config.metrics_jsonl_path)
            if getattr(self.training_config, "metrics_jsonl_path", "")
            else self.output_dir / "metrics.jsonl"
        )
        self.metrics_csv_path = (
            resolve_path(self.training_config.metrics_csv_path)
            if getattr(self.training_config, "metrics_csv_path", "")
            else self.output_dir / "metrics.csv"
        )
        self.sample_prompts_path = str(getattr(self.training_config, "sample_prompts_path", "") or "").strip()
        self.sample_output_dir: Path | None = None
        self.sample_prompts: list[dict[str, Any]] = []
        self.sample_tokenizer: SentencePieceTokenizer | None = None

        if self.sample_prompts_path:
            self.sample_prompts = load_eval_prompts(self.sample_prompts_path)
            sample_dir = (
                resolve_path(self.training_config.sample_output_dir)
                if getattr(self.training_config, "sample_output_dir", "")
                else self.output_dir / "sample_generations"
            )
            self.sample_output_dir = ensure_dir(sample_dir)
            self.sample_tokenizer = SentencePieceTokenizer(self.tokenizer_model_path)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.bad_eval_count = 0
        self.last_val_loss = float("inf")

    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, loss = self.model(x, y)
        if loss is None:
            raise RuntimeError("Model returned no loss.")
        coefficient = float(getattr(self.training_config, "z_loss_coefficient", 0.0))
        if coefficient > 0.0:
            valid = y.ne(-100)
            if valid.any():
                log_z = torch.logsumexp(logits.float(), dim=-1)
                z_loss = log_z[valid].square().mean()
                loss = loss + (coefficient * z_loss)
        return loss

    def _autocast_context(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _checkpoint_state(self) -> dict[str, Any]:
        raw_model = unwrap_model(self.model)
        return {
            "task_name": self.task_name,
            "model_state": raw_model.state_dict(),
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
            "model_config_hash": stable_json_hash(self.model_config),
            "tokenizer_sha256": sha256_file(self.tokenizer_model_path),
        }

    def save_checkpoint(self, path: str | Path, label: str) -> Path:
        checkpoint_path = save_torch_checkpoint(self._checkpoint_state(), path)
        print(f"[checkpoint] saved {label}: {checkpoint_path}")
        return checkpoint_path

    def _append_metrics_row(self, row: dict[str, Any]) -> None:
        ensure_parent_dir(self.metrics_jsonl_path)
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        fieldnames = [
            "step",
            "event",
            "task_name",
            "train_loss",
            "val_loss",
            "val_perplexity",
            "lr",
            "tokens_per_sec",
            "best_val_loss",
            "bad_eval_count",
            "router_load_balance_loss",
            "router_z_loss",
        ]
        ensure_parent_dir(self.metrics_csv_path)
        csv_exists = self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            writer.writerow({key: row.get(key) for key in fieldnames})

    def _record_metrics(
        self,
        event: str,
        train_loss: float | None = None,
        val_loss: float | None = None,
        lr: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        raw_model = unwrap_model(self.model)
        router_load_balance = getattr(
            raw_model,
            "last_router_load_balance_loss",
            None,
        )
        router_z = getattr(raw_model, "last_router_z_loss", None)
        row = {
            "step": self.global_step,
            "event": event,
            "task_name": self.task_name,
            "train_loss": None if train_loss is None else round(float(train_loss), 6),
            "val_loss": None if val_loss is None else round(float(val_loss), 6),
            "val_perplexity": None if val_loss is None else round(math.exp(min(float(val_loss), 20.0)), 6),
            "lr": None if lr is None else float(lr),
            "tokens_per_sec": None if tokens_per_sec is None else round(float(tokens_per_sec), 2),
            "best_val_loss": round(float(self.best_val_loss), 6) if math.isfinite(self.best_val_loss) else None,
            "bad_eval_count": int(self.bad_eval_count),
            "router_load_balance_loss": (
                None
                if router_load_balance is None
                else round(float(router_load_balance.cpu().item()), 6)
            ),
            "router_z_loss": (
                None
                if router_z is None
                else round(float(router_z.cpu().item()), 6)
            ),
        }
        self._append_metrics_row(row)

    def _save_eval_samples(self) -> None:
        if not self.sample_prompts or self.sample_output_dir is None or self.sample_tokenizer is None:
            return

        summary, samples = run_generation_eval(
            model=self.model,
            tokenizer=self.sample_tokenizer,
            prompts=self.sample_prompts,
            device=self.device,
            max_new_tokens=int(self.training_config.sample_max_new_tokens),
            temperature=float(self.training_config.sample_temperature),
            top_k=int(self.training_config.sample_top_k),
            repetition_penalty=float(self.training_config.sample_repetition_penalty),
            default_system_prompt=str(self.training_config.sample_system_prompt),
            max_history_turns=int(self.training_config.sample_max_history_turns),
            default_json_mode=bool(self.training_config.sample_json_mode),
        )
        payload = {
            "task_name": self.task_name,
            "step": self.global_step,
            "summary": summary,
            "samples": samples,
        }
        out_path = self.sample_output_dir / f"step_{self.global_step:07d}.json"
        write_json(payload, out_path)
        print(f"[sample] wrote {len(samples)} sample generations: {out_path}")

    def load_checkpoint(self, path: str | Path) -> None:
        state = load_torch_checkpoint(path, map_location="cpu")
        verify_checkpoint_fingerprints(
            state,
            model_config=self.model_config,
            tokenizer_model_path=self.tokenizer_model_path,
            context="resume checkpoint",
        )

        required = ["model_state", "optimizer_state", "scheduler_state", "global_step", "model_config"]
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(
                f"Checkpoint is missing required fields {missing}: {Path(path)}"
            )

        unwrap_model(self.model).load_state_dict(state["model_state"])
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
                loss = self._forward_loss(x, y)
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
        print(f"[train] metrics_jsonl={self.metrics_jsonl_path}")
        print(f"[train] metrics_csv={self.metrics_csv_path}")
        if self.sample_prompts:
            print(
                f"[train] sample_prompts={self.sample_prompts_path} "
                f"-> {self.sample_output_dir}"
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
                    loss = self._forward_loss(x, y)
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
                self._record_metrics(
                    event="train",
                    train_loss=avg_loss,
                    lr=lr,
                    tokens_per_sec=toks_per_sec,
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
                current_lr = self.optimizer.param_groups[0]["lr"]

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

                self._record_metrics(
                    event="eval",
                    val_loss=self.last_val_loss,
                    lr=current_lr,
                )
                self._save_eval_samples()

                if self.training_config.patience > 0 and self.bad_eval_count >= self.training_config.patience:
                    self.save_checkpoint(self.latest_path, "latest")
                    snapshot = (
                        self.output_dir
                        / f"step_{self.global_step:07d}.safetensors"
                    )
                    self.save_checkpoint(snapshot, f"snapshot@{self.global_step}")
                    print("[early-stop] patience exhausted. Stopping training.")
                    return

            if should_save:
                self.save_checkpoint(self.latest_path, "latest")
                snapshot = (
                    self.output_dir
                    / f"step_{self.global_step:07d}.safetensors"
                )
                self.save_checkpoint(snapshot, f"snapshot@{self.global_step}")

        print("[train] finished max_steps.")
