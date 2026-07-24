from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.config import ModelConfig
from src.generation import build_chat_prompt_tokens
from src.model import GPT
from tests._helpers import train_test_tokenizer


class ModelAndGenerationTests(unittest.TestCase):
    def test_gpt_forward_returns_logits_and_loss(self) -> None:
        cfg = ModelConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            mlp_ratio=2,
            dropout=0.0,
        )
        model = GPT(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), dtype=torch.long)
        targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), dtype=torch.long)

        logits, loss = model(input_ids, targets)

        self.assertEqual(logits.shape, (2, cfg.block_size, cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_gpt_supports_rope_rmsnorm_swiglu_and_checkpointing(self) -> None:
        cfg = ModelConfig(
            vocab_size=32,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_kv_head=1,
            n_embd=16,
            mlp_ratio=2,
            dropout=0.0,
            norm_type="rmsnorm",
            mlp_type="swiglu",
            positional_embedding="rope",
            attention_impl="auto",
            gradient_checkpointing=True,
        )
        model = GPT(cfg)
        model.train()
        input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), dtype=torch.long)
        targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size), dtype=torch.long)

        logits, loss = model(input_ids, targets)

        self.assertEqual(logits.shape, (2, cfg.block_size, cfg.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            model.blocks[0].attn.qkv_proj.out_features,
            cfg.n_embd + 2 * (cfg.n_kv_head * (cfg.n_embd // cfg.n_head)),
        )

    def test_chat_prompt_builder_keeps_prompt_within_context_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = train_test_tokenizer(Path(tmp))
            prompt = build_chat_prompt_tokens(
                tokenizer=tokenizer,
                system_prompt="You explain things clearly.",
                history=[
                    ("What is tokenization?", "It breaks text into tokens."),
                    ("Why use validation?", "It helps detect overfitting."),
                    ("What is SFT?", "It teaches response formatting."),
                ],
                user_message="Give me a JSON summary.",
                block_size=40,
                max_history_turns=2,
                json_mode=True,
            )

            self.assertLessEqual(len(prompt), 39)
            self.assertEqual(prompt[0], tokenizer.bos_id)
            self.assertIn(tokenizer.token_to_id("<|json|>"), prompt)
