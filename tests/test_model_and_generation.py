from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.chat_format import encode_conversation
from src.config import ModelConfig
from src.generation import (
    build_chat_prompt_tokens,
    build_messages_prompt_tokens,
    generate,
)
from src.model import GPT
from src.moe import SparseMoE
from tests._helpers import train_test_tokenizer


class _RecordingAutoregressiveModel(torch.nn.Module):
    def __init__(self, vocab_size: int, block_size: int, next_token_id: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"block_size": block_size})()
        self.vocab_size = vocab_size
        self.next_token_id = next_token_id
        self.contexts: list[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        self.contexts.append(input_ids.detach().clone())
        batch, time = input_ids.shape
        logits = torch.zeros((batch, time, self.vocab_size), dtype=torch.float32)
        logits[:, -1, self.next_token_id] = 1000.0
        return logits, None


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
        self.assertIsInstance(model.blocks[0].mlp, SparseMoE)
        self.assertIsNotNone(model.last_router_load_balance_loss)
        self.assertIsNotNone(model.last_router_z_loss)
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
                block_size=128,
                max_history_turns=2,
                json_mode=True,
            )

            self.assertLessEqual(len(prompt), 127)
            self.assertEqual(prompt[0], tokenizer.bos_id)
            self.assertIn(tokenizer.token_to_id("<|json|>"), prompt)
            self.assertIn(tokenizer.token_to_id("<|system|>"), prompt)

    def test_message_compaction_preserves_instructions_tools_and_current_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = train_test_tokenizer(Path(tmp))
            prompt = build_messages_prompt_tokens(
                tokenizer,
                [
                    {"role": "system", "content": "Always use the tool result."},
                    {"role": "user", "content": "old question " * 20},
                    {"role": "assistant", "content": "old answer " * 20},
                    {"role": "user", "content": "Current arithmetic objective."},
                    {
                        "role": "assistant",
                        "reasoning_content": "Need exact arithmetic. " * 10,
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "calculator",
                                    "arguments": "{\"expression\":\"19*23\"}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "calculator",
                        "content": "{\"ok\":true,\"result\":437}",
                    },
                ],
                block_size=512,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                thinking_mode=True,
            )

            self.assertLessEqual(len(prompt), 511)
            self.assertEqual(prompt[0], tokenizer.bos_id)
            self.assertIn(tokenizer.token_to_id("<|system|>"), prompt)
            self.assertIn(tokenizer.token_to_id("<|tools|>"), prompt)
            self.assertIn(tokenizer.token_to_id("<|tool_response|>"), prompt)
            self.assertIn(tokenizer.token_to_id("<|assistant|>"), prompt)

    def test_completion_budget_keeps_instruction_and_tool_prefix_for_every_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = train_test_tokenizer(Path(tmp))
            block_size = 512
            max_new_tokens = 32
            prompt = build_messages_prompt_tokens(
                tokenizer,
                [
                    {"role": "system", "content": "Always obey the tool schema."},
                    {"role": "user", "content": "old context " * 100},
                    {"role": "assistant", "content": "old response " * 100},
                    {"role": "user", "content": "Use the calculator now."},
                ],
                block_size=block_size,
                max_new_tokens=max_new_tokens,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
            model = _RecordingAutoregressiveModel(
                tokenizer.vocab_size,
                block_size,
                tokenizer.unk_id,
            )
            input_ids = torch.tensor([prompt], dtype=torch.long)

            output = generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )

            self.assertLessEqual(len(prompt), block_size - max_new_tokens)
            self.assertEqual(output.shape[1], len(prompt) + max_new_tokens)
            self.assertEqual(len(model.contexts), max_new_tokens)
            self.assertIn(tokenizer.token_to_id("<|system|>"), prompt)
            self.assertIn(tokenizer.token_to_id("<|tools|>"), prompt)
            for context in model.contexts:
                self.assertTrue(torch.equal(context[0, : len(prompt)], input_ids[0]))

    def test_json_generation_prefix_matches_sft_control_token_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = train_test_tokenizer(Path(tmp))
            messages = [{"role": "user", "content": "Return JSON."}]
            token_ids = {
                name: tokenizer.token_to_id(name)
                for name in (
                    "<|assistant|>",
                    "<|think|>",
                    "<|final|>",
                    "<|json|>",
                )
            }

            no_thinking = build_messages_prompt_tokens(
                tokenizer,
                messages,
                block_size=128,
                max_new_tokens=8,
                json_mode=True,
                thinking_mode=False,
            )
            no_thinking_controls = [
                token
                for token in no_thinking
                if token in token_ids.values()
            ]
            expected_no_thinking_suffix = tokenizer.encode(
                "<|assistant|>\n<|final|>\n<|json|>\n",
                add_bos=False,
                add_eos=False,
            )
            self.assertEqual(
                no_thinking[-len(expected_no_thinking_suffix) :],
                expected_no_thinking_suffix,
            )
            self.assertEqual(
                no_thinking_controls[-3:],
                [
                    token_ids["<|assistant|>"],
                    token_ids["<|final|>"],
                    token_ids["<|json|>"],
                ],
            )

            thinking = build_messages_prompt_tokens(
                tokenizer,
                messages,
                block_size=128,
                max_new_tokens=8,
                json_mode=True,
                thinking_mode=True,
            )
            thinking_controls = [
                token
                for token in thinking
                if token in token_ids.values()
            ]
            expected_thinking_suffix = tokenizer.encode(
                "<|assistant|>\n<|think|>\n",
                add_bos=False,
                add_eos=False,
            )
            self.assertEqual(
                thinking[-len(expected_thinking_suffix) :],
                expected_thinking_suffix,
            )
            self.assertEqual(
                thinking_controls[-2:],
                [
                    token_ids["<|assistant|>"],
                    token_ids["<|think|>"],
                ],
            )

            sft_tokens, _ = encode_conversation(
                tokenizer,
                [
                    {"role": "user", "content": "Return JSON."},
                    {
                        "role": "assistant",
                        "reasoning_content": "Construct the object.",
                        "content": '<|json|>\n{"ok":true}\n</json>',
                    },
                ],
            )
            sft_controls = [
                token
                for token in sft_tokens
                if token in token_ids.values()
            ]
            self.assertEqual(
                sft_controls[-4:],
                [
                    token_ids["<|assistant|>"],
                    token_ids["<|think|>"],
                    token_ids["<|final|>"],
                    token_ids["<|json|>"],
                ],
            )
