from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import torch

from src.chat_format import encode_conversation, legacy_turns_to_messages
from src.tokenizer_utils import SentencePieceTokenizer


def _normalize_chat_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _encode_role_segment(tokenizer: SentencePieceTokenizer, role_token: str, text: str) -> List[int]:
    text = _normalize_chat_text(text)
    return tokenizer.encode(f"{role_token}\n{text}\n", add_bos=False, add_eos=False)


def build_chat_prompt_tokens(
    tokenizer: SentencePieceTokenizer,
    system_prompt: str,
    history: Sequence[Tuple[str, str]],
    user_message: str,
    block_size: int,
    max_history_turns: int = 4,
    json_mode: bool = False,
    tools: Sequence[dict] | None = None,
    thinking_mode: bool = False,
) -> List[int]:
    if block_size < 8:
        raise ValueError("block_size must be at least 8 for chat prompting.")

    turns = list(history[-max_history_turns:] if max_history_turns > 0 else [])

    def compose(selected_turns: Sequence[Tuple[str, str]]) -> List[int]:
        messages = legacy_turns_to_messages(
            system_prompt,
            selected_turns,
            user_message,
        )
        tokens, _ = encode_conversation(
            tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            json_mode=json_mode,
            thinking_mode=thinking_mode,
        )
        return tokens

    prompt_tokens = compose(turns)

    while len(prompt_tokens) > block_size - 1 and turns:
        turns = turns[1:]
        prompt_tokens = compose(turns)

    if len(prompt_tokens) > block_size - 1:
        tail = prompt_tokens[-(block_size - 1) :]
        if tail[0] != tokenizer.bos_id:
            tail = [tokenizer.bos_id] + tail[-(block_size - 2) :]
        prompt_tokens = tail

    return prompt_tokens


def build_messages_prompt_tokens(
    tokenizer: SentencePieceTokenizer,
    messages: Sequence[dict],
    block_size: int,
    *,
    tools: Sequence[dict] | None = None,
    json_mode: bool = False,
    thinking_mode: bool = False,
) -> List[int]:
    tokens, _ = encode_conversation(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        json_mode=json_mode,
        thinking_mode=thinking_mode,
    )
    if len(tokens) > block_size - 1:
        tokens = [tokenizer.bos_id] + tokens[-(block_size - 2) :]
    return tokens


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"logits must have shape [batch, vocab], got {tuple(logits.shape)}")

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        top_values, _ = torch.topk(logits, top_k)
        kth_value = top_values[:, [-1]]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits
    if token_history.dim() != 2:
        raise ValueError(f"token_history must have shape [batch, time], got {tuple(token_history.shape)}")

    adjusted = logits.clone()
    for batch_idx in range(token_history.size(0)):
        token_ids = torch.unique(token_history[batch_idx])
        selected = adjusted[batch_idx, token_ids]
        adjusted[batch_idx, token_ids] = torch.where(
            selected < 0,
            selected * repetition_penalty,
            selected / repetition_penalty,
        )
    return adjusted


@torch.inference_mode()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    stop_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    model.eval()
    stop_set = set(stop_token_ids or [])

    out = input_ids
    for _ in range(max_new_tokens):
        idx_cond = out[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            idx_cond,
            repetition_penalty=repetition_penalty,
        )
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )
        out = torch.cat([out, next_token], dim=1)

        if stop_set:
            done = [int(token.item()) in stop_set for token in next_token]
            if all(done):
                break

    return out


@torch.inference_mode()
def generate_stream(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    stop_token_ids: Sequence[int] | None = None,
) -> Iterator[torch.Tensor]:
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must have shape [batch, time], got {tuple(input_ids.shape)}")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    model.eval()
    stop_set = set(stop_token_ids or [])

    out = input_ids
    for _ in range(max_new_tokens):
        idx_cond = out[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            idx_cond,
            repetition_penalty=repetition_penalty,
        )
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )

        if stop_set:
            done = [int(token.item()) in stop_set for token in next_token]
            if all(done):
                break

        out = torch.cat([out, next_token], dim=1)
        yield next_token
