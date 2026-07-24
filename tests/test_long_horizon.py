from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.config import ModelConfig
from src.long_horizon import (
    compact_task_trace,
    load_task_state,
    run_long_horizon_task,
)
from src.tools import build_default_tool_registry


class LongHorizonTaskTests(unittest.TestCase):
    def test_task_checkpoints_each_step_and_resumes_to_completion(self) -> None:
        generated = iter(
            [
                (
                    "<|think|>Use exact arithmetic.</|think|>"
                    '<|tool_call|>{"id":"math_1","name":"calculator",'
                    '"arguments":{"expression":"19*23"}}</|tool_call|>'
                ),
                "<|final|>The exact result is 437.",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "task.json"
            common = {
                "state_path": state_path,
                "system_prompt": "Use tools for exact arithmetic.",
                "max_steps": 4,
                "runtime_fingerprint": "runtime-v1",
                "model": None,
                "tokenizer": None,
                "model_cfg": ModelConfig(
                    vocab_size=64,
                    block_size=256,
                    n_layer=1,
                    n_head=2,
                    n_kv_head=1,
                    n_embd=16,
                    mlp_ratio=2,
                ),
                "device": torch.device("cpu"),
                "registry": build_default_tool_registry(),
                "json_mode": False,
                "thinking_mode": True,
                "temperature": 0.0,
                "top_k": 0,
                "max_new_tokens": 64,
                "repetition_penalty": 1.0,
            }
            with patch(
                "src.long_horizon.generate_messages_response",
                side_effect=lambda **_: next(generated),
            ):
                paused = run_long_horizon_task(
                    objective="What is 19 times 23?",
                    steps_per_run=1,
                    **common,
                )
                self.assertEqual(paused.status, "paused")
                self.assertEqual(paused.steps_completed, 1)
                self.assertTrue(state_path.exists())

                completed = run_long_horizon_task(
                    objective="",
                    steps_per_run=1,
                    **common,
                )

            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.final_response, "The exact result is 437.")
            self.assertEqual(len(completed.events), 1)
            reloaded = load_task_state(state_path)
            self.assertEqual(reloaded.steps_completed, 2)

    def test_trace_compaction_preserves_recent_messages_and_ledger(self) -> None:
        from src.long_horizon import LongHorizonTaskState

        state = LongHorizonTaskState(
            task_id="task",
            objective="objective",
            system_prompt="system",
            max_steps=100,
            runtime_fingerprint="runtime",
            messages=[
                {"role": "tool", "name": "calculator", "content": str(index)}
                for index in range(10)
            ],
        )
        compact_task_trace(state, max_trace_messages=4)

        self.assertEqual(len(state.messages), 4)
        self.assertIn("tool calculator: 0", state.working_memory)
        self.assertEqual(state.messages[-1]["content"], "9")
