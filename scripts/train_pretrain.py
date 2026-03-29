from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_pretrain_config
from src.datasets import PretrainNpyDataset
from src.model import GPT
from src.tokenizer_utils import SentencePieceTokenizer
from src.trainer import Trainer
from src.utils import (
    assert_exists,
    count_parameters,
    get_device,
    human_count,
    maybe_compile_model,
    read_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-token pretraining.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_tiny.yaml",
        help="Path to pretraining YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume from. Overrides training.resume_from in the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pretrain_config(args.config)
    resume_path = args.resume or cfg.training.resume_from

    set_seed(cfg.training.seed)

    assert_exists(cfg.data.train_array_path, "Packed train array")
    assert_exists(cfg.data.val_array_path, "Packed validation array")
    assert_exists(cfg.data.meta_path, "Packed data metadata")

    tokenizer = SentencePieceTokenizer(cfg.data.tokenizer_model_path)
    meta = read_json(cfg.data.meta_path)

    meta_vocab = int(meta.get("vocab_size", -1))
    meta_block = int(meta.get("block_size", -1))

    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match model.vocab_size "
            f"({cfg.model.vocab_size})."
        )
    if meta_vocab != cfg.model.vocab_size:
        raise ValueError(
            f"Packed data vocab size ({meta_vocab}) does not match model.vocab_size ({cfg.model.vocab_size})."
        )
    if meta_block != cfg.model.block_size:
        raise ValueError(
            f"Packed data block_size ({meta_block}) does not match model.block_size ({cfg.model.block_size})."
        )

    train_dataset = PretrainNpyDataset(cfg.data.train_array_path, block_size=cfg.model.block_size)
    val_dataset = PretrainNpyDataset(cfg.data.val_array_path, block_size=cfg.model.block_size)

    device = get_device("auto")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = GPT(cfg.model)
    model = maybe_compile_model(
        model,
        enabled=cfg.training.compile_model,
        backend=cfg.training.compile_backend,
        mode=cfg.training.compile_mode,
    )
    print(f"[model] trainable parameters: {human_count(count_parameters(model))}")

    trainer = Trainer(
        model=model,
        model_config=cfg.model,
        training_config=cfg.training,
        tokenizer_model_path=cfg.data.tokenizer_model_path,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        task_name="pretrain",
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()
