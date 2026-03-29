from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.chat_runtime import (
    generate_chat_response,
    load_personas,
    load_session,
    resolve_persona,
    save_session,
)
from src.config import ModelConfig
from tests._helpers import train_test_tokenizer, write_personas


class _FakeAutoregressiveModel(torch.nn.Module):
    def __init__(self, vocab_size: int, block_size: int, sequence: list[int]) -> None:
        super().__init__()
        self.config = type("Config", (), {"block_size": block_size})()
        self.vocab_size = vocab_size
        self.sequence = sequence
        self.step = 0

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, None]:
        batch, time = input_ids.shape
        logits = torch.zeros((batch, time, self.vocab_size), dtype=torch.float32, device=input_ids.device)
        token_id = self.sequence[min(self.step, len(self.sequence) - 1)]
        logits[:, -1, token_id] = 1000.0
        self.step += 1
        return logits, None


class ChatRuntimeTests(unittest.TestCase):
    def test_personas_load_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            personas_path = write_personas(Path(tmp) / "personas.json")

            personas = load_personas(personas_path)
            persona = resolve_persona(personas, "json_bot", fallback_system_prompt="fallback")

            self.assertIn("technical", personas)
            self.assertEqual(persona.name, "json_bot")
            self.assertTrue(persona.json_mode)

    def test_session_round_trip_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_path = Path(tmp) / "sessions" / "chat.json"

            save_session(
                session_path,
                history=[("What is SFT?", "It teaches response behavior.")],
                persona_name="technical",
                system_prompt="You explain things clearly.",
                json_mode=False,
            )
            loaded = load_session(session_path)

            self.assertEqual(loaded["persona"], "technical")
            self.assertEqual(
                loaded["history"],
                [("What is SFT?", "It teaches response behavior.")],
            )

    def test_generate_chat_response_matches_streaming_output_in_json_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = train_test_tokenizer(Path(tmp))
            content_token = tokenizer.encode("topic", add_bos=False, add_eos=False)[0]
            end_json = tokenizer.token_to_id("</json>")
            model_cfg = ModelConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=32,
                n_layer=1,
                n_head=2,
                n_embd=16,
                mlp_ratio=2,
                dropout=0.0,
            )

            plain_model = _FakeAutoregressiveModel(
                vocab_size=tokenizer.vocab_size,
                block_size=model_cfg.block_size,
                sequence=[content_token, end_json],
            )
            streamed_model = _FakeAutoregressiveModel(
                vocab_size=tokenizer.vocab_size,
                block_size=model_cfg.block_size,
                sequence=[content_token, end_json],
            )

            plain = generate_chat_response(
                model=plain_model,
                tokenizer=tokenizer,
                model_cfg=model_cfg,
                device=torch.device("cpu"),
                system_prompt="Return JSON when asked.",
                history=[],
                user_message="Return a JSON topic field.",
                max_history_turns=0,
                json_mode=True,
                temperature=0.0,
                top_k=0,
                max_new_tokens=4,
                repetition_penalty=1.0,
            )

            chunks: list[str] = []
            streamed = generate_chat_response(
                model=streamed_model,
                tokenizer=tokenizer,
                model_cfg=model_cfg,
                device=torch.device("cpu"),
                system_prompt="Return JSON when asked.",
                history=[],
                user_message="Return a JSON topic field.",
                max_history_turns=0,
                json_mode=True,
                temperature=0.0,
                top_k=0,
                max_new_tokens=4,
                repetition_penalty=1.0,
                on_text_chunk=chunks.append,
            )

            self.assertEqual(plain, streamed)
            self.assertEqual("".join(chunks), plain)
            self.assertNotIn("</json>", plain)
