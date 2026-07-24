from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_tokenizer_config
from src.data_utils import (
    append_sft_jsonl_to_corpus,
    collect_markdown_files,
    deduplicate_markdown_files,
    deterministic_train_val_split,
    save_manifest,
    write_combined_corpus,
)
from src.tokenizer_utils import SentencePieceTokenizer, train_sentencepiece_tokenizer
from src.utils import ensure_dir, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer on Markdown train documents.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizer.yaml",
        help="Path to tokenizer YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_tokenizer_config(args.config)

    ensure_dir(cfg.processed_data_dir)

    discovered_raw_files = collect_markdown_files(cfg.raw_data_dir)
    raw_files, duplicate_raw_files = deduplicate_markdown_files(discovered_raw_files)
    train_files, val_files = deterministic_train_val_split(
        raw_files,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )

    save_manifest(train_files, cfg.train_manifest_path)
    save_manifest(val_files, cfg.val_manifest_path)

    train_doc_count = write_combined_corpus(train_files, cfg.train_corpus_path)
    val_doc_count = write_combined_corpus(val_files, cfg.val_corpus_path)
    additional_corpus_stats = {"records": 0, "segments": 0}
    available_additional_paths = [
        path for path in cfg.additional_corpus_jsonl_paths if resolve_path(path).exists()
    ]
    missing_additional_paths = [
        path for path in cfg.additional_corpus_jsonl_paths if not resolve_path(path).exists()
    ]
    if missing_additional_paths and cfg.require_additional_corpus:
        formatted = "\n".join(f"  - {resolve_path(path)}" for path in missing_additional_paths)
        raise FileNotFoundError(
            "Required additional tokenizer corpora are missing. Download public datasets first:\n"
            f"{formatted}"
        )
    if available_additional_paths:
        additional_corpus_stats = append_sft_jsonl_to_corpus(
            available_additional_paths,
            cfg.train_corpus_path,
        )

    model_path, vocab_path = train_sentencepiece_tokenizer(
        input_text_path=cfg.train_corpus_path,
        output_prefix=cfg.tokenizer_prefix,
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        character_coverage=cfg.character_coverage,
        normalization_rule_name=cfg.normalization_rule_name,
        user_defined_symbols=cfg.user_defined_symbols,
        unk_id=cfg.unk_id,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        pad_id=cfg.pad_id,
    )

    tokenizer = SentencePieceTokenizer(model_path)
    meta = {
        "tokenizer_model_path": str(resolve_path(model_path)),
        "tokenizer_vocab_path": str(resolve_path(vocab_path)),
        "requested_vocab_size": cfg.vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "model_type": cfg.model_type,
        "seed": cfg.seed,
        "val_fraction": cfg.val_fraction,
        "discovered_documents": len(discovered_raw_files),
        "duplicate_documents_skipped": len(duplicate_raw_files),
        "unique_documents": len(raw_files),
        "train_documents": len(train_files),
        "val_documents": len(val_files),
        "train_documents_kept_after_normalization": train_doc_count,
        "val_documents_kept_after_normalization": val_doc_count,
        "additional_corpus_jsonl_paths": [
            str(resolve_path(path)) for path in available_additional_paths
        ],
        "additional_corpus_records": additional_corpus_stats["records"],
        "additional_corpus_segments": additional_corpus_stats["segments"],
        "special_tokens": tokenizer.get_special_token_map(),
    }
    write_json(meta, cfg.tokenizer_meta_path)

    print("[done] tokenizer training complete")
    print(f"[info] discovered docs: {len(discovered_raw_files)}")
    print(f"[info] duplicate docs skipped: {len(duplicate_raw_files)}")
    print(f"[info] unique docs: {len(raw_files)}")
    print(f"[info] train docs: {len(train_files)}")
    print(f"[info] val docs:   {len(val_files)}")
    print(f"[info] model:      {resolve_path(model_path)}")
    print(f"[info] vocab:      {resolve_path(vocab_path)}")
    print(f"[info] meta:       {resolve_path(cfg.tokenizer_meta_path)}")
    print(f"[info] vocab size requested: {cfg.vocab_size}")
    print(f"[info] vocab size actual:    {tokenizer.vocab_size}")
    print(
        f"[info] additional SFT corpus: {additional_corpus_stats['records']} records, "
        f"{additional_corpus_stats['segments']} text segments"
    )
    for missing_path in missing_additional_paths:
        print(f"[warning] optional tokenizer corpus not found: {resolve_path(missing_path)}")

    if tokenizer.vocab_size != cfg.vocab_size:
        print(
            "[warning] tokenizer actual vocab size differs from requested size. "
            "If later stages report a vocab mismatch, update model.vocab_size in "
            "configs/pretrain_tiny.yaml and configs/sft_tiny.yaml to match tokenizer_meta.json."
        )


if __name__ == "__main__":
    main()
