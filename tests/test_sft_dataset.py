from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.datasets import SFTJsonlDataset
from tests._helpers import train_test_tokenizer, write_jsonl


class SFTDatasetTests(unittest.TestCase):
    def test_sft_dataset_masks_prompt_tokens_and_keeps_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = train_test_tokenizer(root)
            jsonl_path = write_jsonl(
                root / "sample.jsonl",
                [
                    {
                        "system": "You are a practical assistant.",
                        "user": "Explain tokenization simply.",
                        "assistant": "Tokenization splits text into model-friendly pieces.",
                    },
                    {
                        "system": "",
                        "user": "Return a JSON answer.",
                        "assistant": "<|json|>\n{\"ok\": true}\n</json>",
                    },
                ],
            )

            dataset = SFTJsonlDataset(jsonl_path, tokenizer=tokenizer, block_size=96)
            input_ids, labels = dataset[0]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(input_ids.shape[0], 96)
            self.assertEqual(labels.shape[0], 96)
            self.assertIn(-100, labels.tolist())
            self.assertTrue(any(token >= 0 for token in labels.tolist()))
