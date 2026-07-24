from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.config import ModelConfig
from src.evaluation import load_eval_prompts, run_generation_eval
from src.model import GPT
from tests._helpers import train_test_tokenizer, write_eval_prompts


class EvaluationTests(unittest.TestCase):
    def test_generation_eval_loads_prompts_and_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = train_test_tokenizer(root)
            prompts_path = write_eval_prompts(root / "eval_prompts.jsonl")
            prompts = load_eval_prompts(prompts_path)

            model_cfg = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=128,
                n_layer=1,
                n_head=2,
                n_embd=16,
                mlp_ratio=2,
                dropout=0.0,
            )
            model = GPT(model_cfg)

            summary, samples = run_generation_eval(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                device=torch.device("cpu"),
                max_new_tokens=8,
                temperature=0.0,
                top_k=0,
                default_system_prompt="You explain things clearly.",
                max_history_turns=0,
            )

            self.assertEqual(summary["num_prompts"], 2)
            self.assertEqual(len(samples), 2)
            self.assertIn("avg_response_chars", summary)
            self.assertIn("response", samples[0])
