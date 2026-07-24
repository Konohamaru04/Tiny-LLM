from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from src.config import load_chat_config, load_pretrain_config, load_tokenizer_config


class ConfigLoadingTests(unittest.TestCase):
    def test_tokenizer_config_loads_and_derives_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "tokenizer.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    raw_data_dir: data/raw
                    processed_data_dir: data/processed
                    tokenizer_prefix: data/processed/test_tokenizer
                    vocab_size: 64
                    seed: 7
                    val_fraction: 0.2
                    """
                ).strip(),
                encoding="utf-8",
            )

            cfg = load_tokenizer_config(cfg_path)

            self.assertEqual(cfg.vocab_size, 64)
            self.assertEqual(cfg.seed, 7)
            self.assertTrue(cfg.tokenizer_model_path.endswith("test_tokenizer.model"))
            self.assertTrue(cfg.tokenizer_vocab_path.endswith("test_tokenizer.vocab"))
            self.assertTrue(cfg.tokenizer_meta_path.endswith("test_tokenizer_meta.json"))

    def test_pretrain_config_rejects_incompatible_attention_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "pretrain.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      vocab_size: 64
                      block_size: 32
                      n_layer: 2
                      n_head: 3
                      n_embd: 32
                    training:
                      max_steps: 1
                      eval_interval: 1
                      eval_steps: 1
                      save_interval: 1
                      log_interval: 1
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_pretrain_config(cfg_path)

    def test_chat_config_rejects_negative_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "chat.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    checkpoint_path: checkpoints/sft_tiny/best.pt
                    tokenizer_model_path: data/processed/tokenizer.model
                    max_history_turns: 2
                    temperature: -0.1
                    top_k: 40
                    max_new_tokens: 32
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_chat_config(cfg_path)

    def test_pretrain_config_rejects_invalid_grouped_query_attention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "pretrain.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    model:
                      vocab_size: 64
                      block_size: 32
                      n_layer: 2
                      n_head: 6
                      n_kv_head: 4
                      n_embd: 48
                    training:
                      max_steps: 1
                      eval_interval: 1
                      eval_steps: 1
                      save_interval: 1
                      log_interval: 1
                    """
                ).strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "n_kv_head"):
                load_pretrain_config(cfg_path)
