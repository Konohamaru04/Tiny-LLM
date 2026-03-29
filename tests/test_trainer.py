from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ModelConfig, TrainConfig
from src.model import GPT
from src.trainer import Trainer
from tests._helpers import train_test_tokenizer, write_eval_prompts


class TrainerTests(unittest.TestCase):
    def _make_loader(self, vocab_size: int, block_size: int) -> DataLoader:
        x = torch.randint(0, vocab_size, (4, block_size), dtype=torch.long)
        y = torch.randint(0, vocab_size, (4, block_size), dtype=torch.long)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=2, shuffle=False)

    def test_trainer_can_evaluate_train_and_reload_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer = train_test_tokenizer(root)
            tokenizer_path = tokenizer.model_path
            prompts_path = write_eval_prompts(root / "eval_prompts.jsonl")

            model_cfg = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                mlp_ratio=2,
                dropout=0.0,
            )
            train_cfg = TrainConfig(
                output_dir=str(root / "checkpoints"),
                batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                min_lr=1e-4,
                warmup_steps=0,
                max_steps=1,
                num_workers=0,
                weight_decay=0.0,
                grad_clip=1.0,
                eval_interval=1,
                eval_steps=1,
                save_interval=1,
                log_interval=1,
                patience=2,
                use_amp=False,
                sample_prompts_path=str(prompts_path),
                sample_output_dir=str(root / "sample_generations"),
                sample_max_new_tokens=8,
                sample_top_k=0,
            )

            train_loader = self._make_loader(model_cfg.vocab_size, model_cfg.block_size)
            val_loader = self._make_loader(model_cfg.vocab_size, model_cfg.block_size)

            trainer = Trainer(
                model=GPT(model_cfg),
                model_config=model_cfg,
                training_config=train_cfg,
                tokenizer_model_path=tokenizer_path,
                train_loader=train_loader,
                val_loader=val_loader,
                device=torch.device("cpu"),
                task_name="unit-test",
            )

            val_loss = trainer.evaluate()
            self.assertGreaterEqual(val_loss, 0.0)

            trainer.train()

            latest_path = root / "checkpoints" / "latest.pt"
            self.assertTrue(latest_path.exists())
            self.assertTrue((root / "checkpoints" / "metrics.jsonl").exists())
            self.assertTrue((root / "checkpoints" / "metrics.csv").exists())
            self.assertTrue((root / "sample_generations" / "step_0000001.json").exists())

            restored = Trainer(
                model=GPT(model_cfg),
                model_config=model_cfg,
                training_config=train_cfg,
                tokenizer_model_path=tokenizer_path,
                train_loader=self._make_loader(model_cfg.vocab_size, model_cfg.block_size),
                val_loader=self._make_loader(model_cfg.vocab_size, model_cfg.block_size),
                device=torch.device("cpu"),
                task_name="unit-test",
            )
            restored.load_checkpoint(latest_path)

            self.assertEqual(restored.global_step, 1)

    def test_trainer_rejects_tokenizer_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_path = root / "tokenizer_a.model"
            tokenizer_path.write_text("tokenizer-a", encoding="utf-8")
            other_tokenizer_path = root / "tokenizer_b.model"
            other_tokenizer_path.write_text("tokenizer-b", encoding="utf-8")

            model_cfg = ModelConfig(
                vocab_size=32,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                mlp_ratio=2,
                dropout=0.0,
            )
            train_cfg = TrainConfig(
                output_dir=str(root / "checkpoints"),
                batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                min_lr=1e-4,
                warmup_steps=0,
                max_steps=1,
                num_workers=0,
                weight_decay=0.0,
                grad_clip=1.0,
                eval_interval=1,
                eval_steps=1,
                save_interval=1,
                log_interval=1,
                patience=2,
                use_amp=False,
            )

            trainer = Trainer(
                model=GPT(model_cfg),
                model_config=model_cfg,
                training_config=train_cfg,
                tokenizer_model_path=tokenizer_path,
                train_loader=self._make_loader(model_cfg.vocab_size, model_cfg.block_size),
                val_loader=self._make_loader(model_cfg.vocab_size, model_cfg.block_size),
                device=torch.device("cpu"),
                task_name="fingerprint-test",
            )
            checkpoint_path = trainer.save_checkpoint(root / "checkpoints" / "manual.pt", "manual")

            mismatched = Trainer(
                model=GPT(model_cfg),
                model_config=model_cfg,
                training_config=train_cfg,
                tokenizer_model_path=other_tokenizer_path,
                train_loader=self._make_loader(model_cfg.vocab_size, model_cfg.block_size),
                val_loader=self._make_loader(model_cfg.vocab_size, model_cfg.block_size),
                device=torch.device("cpu"),
                task_name="fingerprint-test",
            )

            with self.assertRaises(ValueError):
                mismatched.load_checkpoint(checkpoint_path)
