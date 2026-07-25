from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from src.tokenizer_utils import SentencePieceTokenizer


ROLE_TOKENS = {
    "system": "<|system|>",
    "developer": "<|developer|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}


def assistant_generation_prefix(*, json_mode: bool, thinking_mode: bool) -> str:
    """Return the canonical assistant prefix used for inference.

    JSON marks the final-answer payload, so it follows ``<|final|>`` just as it
    does in supervised assistant targets. Thinking starts first when enabled;
    the model then emits ``<|final|>`` and the optional JSON marker after the
    reasoning block.
    """
    prefix = "<|assistant|>\n"
    if thinking_mode:
        return prefix + "<|think|>\n"
    prefix += "<|final|>\n"
    if json_mode:
        prefix += "<|json|>\n"
    return prefix


def normalize_chat_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_tool_call(call: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    function = call.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name", "")).strip()
        arguments = function.get("arguments", {})
    else:
        name = str(call.get("name", "")).strip()
        arguments = call.get("arguments", {})

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, Mapping):
        arguments = {"value": arguments}

    call_id = str(call.get("id", "")).strip() or f"call_{index + 1}"
    return {
        "id": call_id,
        "name": name,
        "arguments": dict(arguments),
    }


def _encode(
    tokenizer: SentencePieceTokenizer,
    text: str,
    *,
    supervised: bool,
) -> tuple[list[int], list[bool]]:
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    return token_ids, [supervised] * len(token_ids)


def _append_segment(
    tokenizer: SentencePieceTokenizer,
    tokens: list[int],
    loss_mask: list[bool],
    text: str,
    *,
    supervised: bool,
) -> None:
    segment_tokens, segment_mask = _encode(tokenizer, text, supervised=supervised)
    tokens.extend(segment_tokens)
    loss_mask.extend(segment_mask)


def _tool_payload(message: Mapping[str, Any]) -> str:
    payload = {
        "tool_call_id": str(message.get("tool_call_id", "")).strip(),
        "name": str(message.get("name", "")).strip(),
        "content": normalize_chat_text(message.get("content")),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def encode_conversation(
    tokenizer: SentencePieceTokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    add_generation_prompt: bool = False,
    json_mode: bool = False,
    thinking_mode: bool = False,
) -> tuple[list[int], list[bool]]:
    """Encode one canonical chat conversation and its assistant-only loss mask."""
    tokens = [tokenizer.bos_id]
    loss_mask = [False]
    tools_written = False
    has_instruction_role = any(
        str(message.get("role", "")).strip().lower() in {"system", "developer"}
        for message in messages
    )
    if tools and not has_instruction_role:
        tools_json = json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))
        _append_segment(
            tokenizer,
            tokens,
            loss_mask,
            f"<|tools|>\n{tools_json}\n</|tools|>\n",
            supervised=False,
        )
        tools_written = True

    for message_index, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == "environment":
            role = "tool"

        if role in ROLE_TOKENS:
            _append_segment(
                tokenizer,
                tokens,
                loss_mask,
                f"{ROLE_TOKENS[role]}\n",
                supervised=False,
            )
            content = normalize_chat_text(message.get("content"))
            if content and role != "assistant":
                _append_segment(
                    tokenizer,
                    tokens,
                    loss_mask,
                    content + "\n",
                    supervised=False,
                )

            if role in {"system", "developer"} and tools and not tools_written:
                tools_json = json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))
                _append_segment(
                    tokenizer,
                    tokens,
                    loss_mask,
                    f"<|tools|>\n{tools_json}\n</|tools|>\n",
                    supervised=False,
                )
                tools_written = True

            if role == "assistant":
                reasoning = normalize_chat_text(
                    message.get("reasoning_content", message.get("reasoning"))
                )
                if reasoning:
                    _append_segment(
                        tokenizer,
                        tokens,
                        loss_mask,
                        f"<|think|>\n{reasoning}\n</|think|>\n",
                        supervised=True,
                    )

                raw_calls = message.get("tool_calls") or []
                if not isinstance(raw_calls, list):
                    raise ValueError("assistant.tool_calls must be a list")
                for call_index, call in enumerate(raw_calls):
                    if not isinstance(call, Mapping):
                        raise ValueError("Each assistant tool call must be an object")
                    canonical_call = normalize_tool_call(call, call_index)
                    call_json = json.dumps(
                        canonical_call,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    _append_segment(
                        tokenizer,
                        tokens,
                        loss_mask,
                        f"<|tool_call|>\n{call_json}\n</|tool_call|>\n",
                        supervised=True,
                    )

                if content:
                    _append_segment(
                        tokenizer,
                        tokens,
                        loss_mask,
                        f"<|final|>\n{content}\n",
                        supervised=True,
                    )

                next_role = ""
                if message_index + 1 < len(messages):
                    next_role = str(messages[message_index + 1].get("role", "")).lower()
                if not raw_calls or next_role not in {"tool", "environment"}:
                    tokens.append(tokenizer.eos_id)
                    loss_mask.append(True)
            continue

        if role == "tool":
            _append_segment(
                tokenizer,
                tokens,
                loss_mask,
                f"<|tool_response|>\n{_tool_payload(message)}\n</|tool_response|>\n",
                supervised=False,
            )
            continue

        raise ValueError(f"Unsupported chat role: {role!r}")

    if tools and not tools_written:
        tools_json = json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))
        _append_segment(
            tokenizer,
            tokens,
            loss_mask,
            f"<|tools|>\n{tools_json}\n</|tools|>\n",
            supervised=False,
        )

    if add_generation_prompt:
        _append_segment(
            tokenizer,
            tokens,
            loss_mask,
            assistant_generation_prefix(
                json_mode=json_mode,
                thinking_mode=thinking_mode,
            ),
            supervised=False,
        )

    return tokens, loss_mask


def legacy_turns_to_messages(
    system_prompt: str,
    history: Iterable[tuple[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if normalize_chat_text(system_prompt):
        messages.append({"role": "system", "content": normalize_chat_text(system_prompt)})
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": normalize_chat_text(user_text)})
        messages.append({"role": "assistant", "content": normalize_chat_text(assistant_text)})
    messages.append({"role": "user", "content": normalize_chat_text(user_message)})
    return messages
