from __future__ import annotations

from typing import Any, Iterator, List, Mapping, Sequence, Tuple

import torch

from src.capabilities import ThinkingMode, ToolDefinition, build_capability_prefix
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
    thinking_mode: str | ThinkingMode = ThinkingMode.OFF,
    tools: Sequence[ToolDefinition | Mapping[str, Any]] = (),
) -> List[int]:
    if block_size < 8:
        raise ValueError("block_size must be at least 8 for chat prompting.")

    user_message = _normalize_chat_text(user_message)
    system_prompt = _normalize_chat_text(system_prompt)
    turns = list(history[-max_history_turns:] if max_history_turns > 0 else [])

    bos = [tokenizer.bos_id]
    system_tokens = (
        _encode_role_segment(tokenizer, "<|system|>", system_prompt)
        if system_prompt
        else []
    )
    assistant_prefix = tokenizer.encode("<|assistant|>\n", add_bos=False, add_eos=False)
    capability_prefix = tokenizer.encode(
        build_capability_prefix(thinking_mode=thinking_mode, tools=tools),
        add_bos=False,
        add_eos=False,
    )
    json_prefix = (
        tokenizer.encode("<|json|>\n", add_bos=False, add_eos=False)
        if json_mode
        else []
    )
    response_prefix = assistant_prefix + capability_prefix + json_prefix

    history_segments = [
        _encode_role_segment(tokenizer, "<|user|>", user_text)
        + _encode_role_segment(tokenizer, "<|assistant|>", assistant_text)
        for user_text, assistant_text in turns
    ]
    user_header = tokenizer.encode("<|user|>\n", add_bos=False, add_eos=False)
    user_body = tokenizer.encode(f"{user_message}\n", add_bos=False, add_eos=False)

    def compose(selected_history: Sequence[List[int]], selected_user_body: Sequence[int]) -> List[int]:
        tokens = bos + system_tokens
        for segment in selected_history:
            tokens.extend(segment)
        tokens.extend(user_header)
        tokens.extend(selected_user_body)
        tokens.extend(response_prefix)
        return tokens

    prompt_tokens = compose(history_segments, user_body)
    while len(prompt_tokens) > block_size - 1 and history_segments:
        history_segments = history_segments[1:]
        prompt_tokens = compose(history_segments, user_body)

    if len(prompt_tokens) > block_size - 1:
        fixed_tokens = bos + system_tokens + user_header + response_prefix
        available_user_tokens = (block_size - 1) - len(fixed_tokens)
        if available_user_tokens < 0:
            raise ValueError(
                "System prompt, role tokens, capability mode, and tool schemas exceed "
                "the model context window. Reduce the system prompt or tool definitions."
            )
        user_tail = user_body[-available_user_tokens:] if available_user_tokens else []
        prompt_tokens = compose([], user_tail)

    if len(prompt_tokens) > block_size - 1:
        raise RuntimeError("Prompt compaction failed to fit the configured context window")
    return prompt_tokens


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
    if top_k is not None and 0 < top_k < logits.size(-1):
        top_values, _ = torch.topk(logits, top_k)
        kth_value = top_values[:, [-1]]
        logits = torch.where(logits < kth_value, torch.full_like(logits, float("-inf")), logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits
    if token_history.dim() != 2:
        raise ValueError("token_history must have shape [batch, time]")
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
        raise ValueError("input_ids must have shape [batch, time]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    model.eval()
    stop_set = set(stop_token_ids or [])
    out = input_ids
    for _ in range(max_new_tokens):
        idx_cond = out[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        next_token_logits = apply_repetition_penalty(
            logits[:, -1, :], idx_cond, repetition_penalty
        )
        next_token = sample_next_token(next_token_logits, temperature, top_k)
        out = torch.cat([out, next_token], dim=1)
        if stop_set and all(int(token.item()) in stop_set for token in next_token):
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
        raise ValueError("input_ids must have shape [batch, time]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    model.eval()
    stop_set = set(stop_token_ids or [])
    out = input_ids
    for _ in range(max_new_tokens):
        idx_cond = out[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        next_token_logits = apply_repetition_penalty(
            logits[:, -1, :], idx_cond, repetition_penalty
        )
        next_token = sample_next_token(next_token_logits, temperature, top_k)
        if stop_set and all(int(token.item()) in stop_set for token in next_token):
            break
        out = torch.cat([out, next_token], dim=1)
        yield next_token
