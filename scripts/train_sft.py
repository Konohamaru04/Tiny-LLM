from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, load_sft_config
from src.datasets import SFTJsonlDataset
from src.model import GPT
from src.tokenizer_utils import SentencePieceTokenizer
from src.trainer import Trainer
from src.utils import (
    assert_exists,
    count_parameters,
    get_device,
    human_count,
    load_torch_checkpoint,
    maybe_compile_model,
    set_seed,
    verify_checkpoint_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight chat SFT.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_tiny.yaml",
        help="Path to SFT YAML config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional SFT checkpoint path to resume from. Overrides training.resume_from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_sft_config(args.config)
    resume_path = args.resume or cfg.training.resume_from

    set_seed(cfg.training.seed)

    tokenizer = SentencePieceTokenizer(cfg.data.tokenizer_model_path)
    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match model.vocab_size "
            f"({cfg.model.vocab_size})."
        )

    assert_exists(cfg.data.train_jsonl_path, "SFT train JSONL")
    assert_exists(cfg.data.val_jsonl_path, "SFT validation JSONL")

    train_dataset = SFTJsonlDataset(
        cfg.data.train_jsonl_path,
        tokenizer=tokenizer,
        block_size=cfg.model.block_size,
    )
    val_dataset = SFTJsonlDataset(
        cfg.data.val_jsonl_path,
        tokenizer=tokenizer,
        block_size=cfg.model.block_size,
    )

    device = get_device("auto")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
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

    if resume_path:
        print(f"[resume] will resume SFT from checkpoint: {resume_path}")
    else:
        pretrained_ckpt_path = assert_exists(cfg.training.pretrained_checkpoint, "Pretrained checkpoint")
        pretrained_state = load_torch_checkpoint(pretrained_ckpt_path, map_location="cpu")

        ckpt_model_cfg = pretrained_state.get("model_config")
        if ckpt_model_cfg is None:
            raise ValueError(f"Pretrained checkpoint missing model_config: {pretrained_ckpt_path}")

        loaded_cfg = ModelConfig(**ckpt_model_cfg)
        if loaded_cfg != cfg.model:
            raise ValueError(
                "SFT model config does not match the model config stored in the pretrained checkpoint.\n"
                f"SFT config:      {cfg.model}\n"
                f"Checkpoint cfg:  {loaded_cfg}"
            )

        verify_checkpoint_fingerprints(
            pretrained_state,
            model_config=cfg.model,
            tokenizer_model_path=cfg.data.tokenizer_model_path,
            context="pretrained checkpoint",
        )
        model.load_state_dict(pretrained_state["model_state"])
        print(f"[init] loaded pretrained weights from: {pretrained_ckpt_path}")

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
        task_name="sft",
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()
