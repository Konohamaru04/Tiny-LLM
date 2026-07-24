from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_utils import (
    collect_markdown_files,
    deduplicate_markdown_files,
    deterministic_train_val_split,
    encode_documents,
    load_manifest,
    pack_token_stream,
    save_manifest,
)
from tests._helpers import train_test_tokenizer, write_markdown_docs


class DataPipelineTests(unittest.TestCase):
    def test_markdown_content_deduplication_keeps_one_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            nested_dir = raw_dir / "public"
            nested_dir.mkdir(parents=True)
            (raw_dir / "a.md").write_bytes(b"same\r\ncontent\n")
            (nested_dir / "copy.md").write_bytes(b"same\ncontent\n")
            (nested_dir / "unique.md").write_text("different", encoding="utf-8")

            unique, duplicates = deduplicate_markdown_files(
                collect_markdown_files(raw_dir)
            )

            self.assertEqual(len(unique), 2)
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(unique[0].name, "a.md")
            self.assertEqual(duplicates[0].name, "copy.md")

    def test_markdown_split_manifest_encode_and_pack_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = write_markdown_docs(root)
            tokenizer = train_test_tokenizer(root)

            files = collect_markdown_files(raw_dir)
            train_files, val_files = deterministic_train_val_split(files, val_fraction=0.34, seed=3)

            manifest_path = root / "train_manifest.json"
            save_manifest(train_files, manifest_path)
            loaded_files = load_manifest(manifest_path)

            self.assertEqual(loaded_files, train_files)
            self.assertEqual(len(train_files), 2)
            self.assertEqual(len(val_files), 1)

            token_stream = encode_documents(train_files, tokenizer, add_bos=True, add_eos=True)
            packed, meta = pack_token_stream(token_stream * 3, block_size=16)

            self.assertEqual(packed.ndim, 2)
            self.assertEqual(packed.shape[1], 17)
            self.assertEqual(meta["block_size"], 16)
            self.assertGreater(meta["num_sequences"], 0)
