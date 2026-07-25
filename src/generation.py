from __future__ import annotations

from typing import Any, Iterator, List, Mapping, Sequence, Tuple

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
    max_new_tokens: int = 1,
) -> List[int]:
    if block_size < 8:
        raise ValueError("block_size must be at least 8 for chat prompting.")

    turns = list(history[-max_history_turns:] if max_history_turns > 0 else [])
    messages = legacy_turns_to_messages(system_prompt, turns, user_message)
    return build_messages_prompt_tokens(
        tokenizer,
        messages,
        block_size,
        max_new_tokens=max_new_tokens,
        tools=tools,
        json_mode=json_mode,
        thinking_mode=thinking_mode,
    )


def _encode_messages_prompt(
    tokenizer: SentencePieceTokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[dict] | None,
    json_mode: bool,
    thinking_mode: bool,
) -> List[int]:
    tokens, _ = encode_conversation(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=True,
        json_mode=json_mode,
        thinking_mode=thinking_mode,
    )
    return tokens


def _trim_message_field_to_fit(
    tokenizer: SentencePieceTokenizer,
    messages: list[dict[str, Any]],
    message_index: int,
    field: str,
    *,
    token_limit: int,
    tools: Sequence[dict] | None,
    json_mode: bool,
    thinking_mode: bool,
) -> bool:
    value = messages[message_index].get(field)
    if not isinstance(value, str) or not value:
        return False

    original = value
    messages[message_index][field] = ""
    empty_tokens = _encode_messages_prompt(
        tokenizer,
        messages,
        tools=tools,
        json_mode=json_mode,
        thinking_mode=thinking_mode,
    )
    if len(empty_tokens) > token_limit:
        return True

    low = 0
    high = len(original)
    while low < high:
        keep = (low + high + 1) // 2
        messages[message_index][field] = original[-keep:]
        candidate = _encode_messages_prompt(
            tokenizer,
            messages,
            tools=tools,
            json_mode=json_mode,
            thinking_mode=thinking_mode,
        )
        if len(candidate) <= token_limit:
            low = keep
        else:
            high = keep - 1
    messages[message_index][field] = original[-low:] if low else ""
    return True


def build_messages_prompt_tokens(
    tokenizer: SentencePieceTokenizer,
    messages: Sequence[dict],
    block_size: int,
    *,
    max_new_tokens: int = 1,
    tools: Sequence[dict] | None = None,
    json_mode: bool = False,
    thinking_mode: bool = False,
) -> List[int]:
    if block_size < 8:
        raise ValueError("block_size must be at least 8 for chat prompting.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    token_limit = block_size - max_new_tokens
    if token_limit <= 0:
        raise ValueError(
            "max_new_tokens must be smaller than block_size so the prompt has "
            "at least one token of context."
        )
    working = [dict(message) for message in messages]
    prompt_tokens = _encode_messages_prompt(
        tokenizer,
        working,
        tools=tools,
        json_mode=json_mode,
        thinking_mode=thinking_mode,
    )
    if len(prompt_tokens) <= token_limit:
        return prompt_tokens

    last_user_index = max(
        (
            index
            for index, message in enumerate(working)
            if str(message.get("role", "")).strip().lower() == "user"
        ),
        default=len(working),
    )
    instruction_end = 0
    while instruction_end < last_user_index:
        role = str(working[instruction_end].get("role", "")).strip().lower()
        if role not in {"system", "developer"}:
            break
        instruction_end += 1

    instructions = working[:instruction_end]
    history = working[instruction_end:last_user_index]
    current_trace = working[last_user_index:]
    while history:
        history.pop(0)
        while history:
            next_role = str(history[0].get("role", "")).strip().lower()
            if next_role == "user":
                break
            history.pop(0)
        candidate = instructions + history + current_trace
        prompt_tokens = _encode_messages_prompt(
            tokenizer,
            candidate,
            tools=tools,
            json_mode=json_mode,
            thinking_mode=thinking_mode,
        )
        if len(prompt_tokens) <= token_limit:
            return prompt_tokens

    working = instructions + current_trace
    current_start = len(instructions)
    trim_fields: list[tuple[int, str]] = []
    for index in range(current_start + 1, len(working)):
        role = str(working[index].get("role", "")).strip().lower()
        if role == "assistant":
            trim_fields.extend(
                (index, field)
                for field in ("reasoning_content", "reasoning", "content")
            )
        elif role in {"tool", "environment"}:
            trim_fields.append((index, "content"))
    if current_start < len(working):
        trim_fields.append((current_start, "content"))

    for message_index, field in trim_fields:
        _trim_message_field_to_fit(
            tokenizer,
            working,
            message_index,
            field,
            token_limit=token_limit,
            tools=tools,
            json_mode=json_mode,
            thinking_mode=thinking_mode,
        )
        prompt_tokens = _encode_messages_prompt(
            tokenizer,
            working,
            tools=tools,
            json_mode=json_mode,
            thinking_mode=thinking_mode,
        )
        if len(prompt_tokens) <= token_limit:
            return prompt_tokens

    raise ValueError(
        "System/developer instructions, tool schemas, role markers, and the "
        "current tool-call structure exceed the model context window."
    )


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
    block_size = int(model.config.block_size)
    if input_ids.size(1) + max_new_tokens > block_size:
        raise ValueError(
            "Prompt plus max_new_tokens exceeds the model context window: "
            f"{input_ids.size(1)} + {max_new_tokens} > {block_size}. "
            "Compact the prompt with the same max_new_tokens budget before generation."
        )

    model.eval()
    stop_set = set(stop_token_ids or [])

    out = input_ids
    for _ in range(max_new_tokens):
        idx_cond = out
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
    block_size = int(model.config.block_size)
    if input_ids.size(1) + max_new_tokens > block_size:
        raise ValueError(
            "Prompt plus max_new_tokens exceeds the model context window: "
            f"{input_ids.size(1)} + {max_new_tokens} > {block_size}. "
            "Compact the prompt with the same max_new_tokens budget before generation."
        )

    model.eval()
    stop_set = set(stop_token_ids or [])

    out = input_ids
    for _ in range(max_new_tokens):
        idx_cond = out
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
