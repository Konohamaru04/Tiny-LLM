from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_pretrain_config
from src.data_utils import encode_documents, load_manifest, pack_token_stream
from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import assert_exists, ensure_parent_dir, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare packed token arrays for pretraining.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_tiny.yaml",
        help="Path to pretraining YAML config.",
    )
    return parser.parse_args()


def save_npy(array: np.ndarray, path: str | Path) -> None:
    path = resolve_path(path)
    ensure_parent_dir(path)
    np.save(path, array)


def main() -> None:
    args = parse_args()
    cfg = load_pretrain_config(args.config)

    tokenizer = SentencePieceTokenizer(cfg.data.tokenizer_model_path)
    assert_exists(cfg.data.train_manifest_path, "Train manifest")
    assert_exists(cfg.data.val_manifest_path, "Validation manifest")

    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match model.vocab_size "
            f"({cfg.model.vocab_size}) in {args.config}. Update the config to match the tokenizer."
        )

    train_files = load_manifest(cfg.data.train_manifest_path)
    val_files = load_manifest(cfg.data.val_manifest_path)

    train_tokens = encode_documents(train_files, tokenizer, add_bos=True, add_eos=True)
    val_tokens = encode_documents(val_files, tokenizer, add_bos=True, add_eos=True)

    train_array, train_meta = pack_token_stream(train_tokens, cfg.model.block_size)
    val_array, val_meta = pack_token_stream(val_tokens, cfg.model.block_size)

    save_npy(train_array, cfg.data.train_array_path)
    save_npy(val_array, cfg.data.val_array_path)

    meta = {
        "tokenizer_model_path": str(resolve_path(cfg.data.tokenizer_model_path)),
        "vocab_size": tokenizer.vocab_size,
        "block_size": cfg.model.block_size,
        "train_documents": len(train_files),
        "val_documents": len(val_files),
        "train": train_meta,
        "val": val_meta,
    }
    write_json(meta, cfg.data.meta_path)

    print("[done] data preparation complete")
    print(f"[info] train array: {resolve_path(cfg.data.train_array_path)} shape={train_array.shape}")
    print(f"[info] val array:   {resolve_path(cfg.data.val_array_path)} shape={val_array.shape}")
    print(f"[info] meta:        {resolve_path(cfg.data.meta_path)}")
    print(f"[info] train sequences: {train_array.shape[0]}")
    print(f"[info] val sequences:   {val_array.shape[0]}")
    print(f"[info] block size:      {cfg.model.block_size}")


if __name__ == "__main__":
    main()