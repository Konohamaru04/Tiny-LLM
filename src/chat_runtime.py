from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import torch

from src.config import ChatConfig, ModelConfig
from src.generation import build_chat_prompt_tokens, generate, generate_stream
from src.model import GPT
from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import (
    assert_exists,
    ensure_parent_dir,
    get_device,
    load_torch_checkpoint,
    read_json,
    resolve_path,
    verify_checkpoint_fingerprints,
    write_json,
)


@dataclass
class Persona:
    name: str
    system_prompt: str
    description: str = ""
    json_mode: bool = False


def load_personas(path: str | Path) -> dict[str, Persona]:
    personas_path = resolve_path(path)
    raw = read_json(personas_path)
    if isinstance(raw, dict):
        items = []
        for name, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError(f"Persona entry `{name}` must be an object.")
            items.append({"name": name, **value})
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Personas file must contain a list or object: {personas_path}")

    personas: dict[str, Persona] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Persona entry must be an object in {personas_path}")
        name = str(item.get("name", "")).strip()
        system_prompt = str(item.get("system_prompt", "")).strip()
        if not name or not system_prompt:
            raise ValueError(f"Persona entry must include non-empty `name` and `system_prompt`: {item}")
        personas[name] = Persona(
            name=name,
            system_prompt=system_prompt,
            description=str(item.get("description", "")).strip(),
            json_mode=bool(item.get("json_mode", False)),
        )

    if not personas:
        raise ValueError(f"No personas loaded from {personas_path}")
    return personas


def resolve_persona(
    personas: dict[str, Persona],
    name: str | None,
    fallback_system_prompt: str,
    fallback_json_mode: bool = False,
) -> Persona:
    if name:
        persona = personas.get(name)
        if persona is None:
            available = ", ".join(sorted(personas))
            raise ValueError(f"Unknown persona `{name}`. Available: {available}")
        return persona
    return Persona(
        name="custom",
        system_prompt=fallback_system_prompt,
        description="Ad hoc system prompt",
        json_mode=fallback_json_mode,
    )


def load_session(path: str | Path) -> dict:
    session_path = resolve_path(path)
    if not session_path.exists():
        return {"history": []}
    data = read_json(session_path)
    if not isinstance(data, dict):
        raise ValueError(f"Session file must contain an object: {session_path}")
    history = data.get("history", [])
    if not isinstance(history, list):
        raise ValueError(f"Session history must be a list: {session_path}")
    normalized_history: list[tuple[str, str]] = []
    for item in history:
        if isinstance(item, dict):
            user = str(item.get("user", "")).strip()
            assistant = str(item.get("assistant", "")).strip()
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user = str(item[0]).strip()
            assistant = str(item[1]).strip()
        else:
            continue
        if user or assistant:
            normalized_history.append((user, assistant))
    data["history"] = normalized_history
    return data


def save_session(
    path: str | Path,
    history: Iterable[tuple[str, str]],
    persona_name: str,
    system_prompt: str,
    json_mode: bool,
) -> Path:
    session_path = resolve_path(path)
    ensure_parent_dir(session_path)
    payload = {
        "persona": persona_name,
        "system_prompt": system_prompt,
        "json_mode": json_mode,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "history": [{"user": user, "assistant": assistant} for user, assistant in history],
    }
    write_json(payload, session_path)
    return session_path


def strip_hidden_stop_tokens(tokenizer: SentencePieceTokenizer, token_ids: list[int]) -> list[int]:
    hidden = {
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
        tokenizer.token_to_id("<|json|>"),
        tokenizer.token_to_id("</json>"),
    }
    out = list(token_ids)
    while out and out[-1] in hidden:
        out.pop()
    return out


def build_stop_ids(tokenizer: SentencePieceTokenizer, json_mode: bool) -> list[int]:
    stop_ids = [
        tokenizer.eos_id,
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
    ]
    if json_mode:
        stop_ids.append(tokenizer.token_to_id("</json>"))
    return stop_ids


def generate_chat_response(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    model_cfg: ModelConfig,
    device: torch.device,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
    *,
    max_history_turns: int,
    json_mode: bool,
    temperature: float,
    top_k: int,
    max_new_tokens: int,
    repetition_penalty: float,
    on_text_chunk: Callable[[str], None] | None = None,
) -> str:
    prompt_tokens = build_chat_prompt_tokens(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        history=history,
        user_message=user_message,
        block_size=model_cfg.block_size,
        max_history_turns=max_history_turns,
        json_mode=json_mode,
    )
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    stop_ids = build_stop_ids(tokenizer, json_mode=json_mode)

    if on_text_chunk is None:
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_ids,
        )
        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        generated_ids = strip_hidden_stop_tokens(tokenizer, generated_ids)
        return tokenizer.decode(generated_ids).strip()

    generated_ids: list[int] = []
    rendered_text = ""
    for next_token in generate_stream(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
        repetition_penalty=repetition_penalty,
        stop_token_ids=stop_ids,
    ):
        generated_ids.extend(next_token[0].tolist())
        decoded = tokenizer.decode(strip_hidden_stop_tokens(tokenizer, generated_ids)).strip()
        if decoded.startswith(rendered_text):
            delta = decoded[len(rendered_text) :]
        else:
            delta = decoded
        if delta:
            on_text_chunk(delta)
        rendered_text = decoded

    return tokenizer.decode(strip_hidden_stop_tokens(tokenizer, generated_ids)).strip()


def load_chat_runtime(
    config: ChatConfig,
    checkpoint_override: str = "",
    tokenizer_override: str = "",
) -> tuple[torch.nn.Module, SentencePieceTokenizer, ModelConfig, torch.device]:
    checkpoint_path = checkpoint_override or config.checkpoint_path
    tokenizer_path = tokenizer_override or config.tokenizer_model_path

    checkpoint_path = assert_exists(checkpoint_path, "Chat checkpoint")
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    device = get_device(config.device)

    state = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if "model_config" not in state:
        raise ValueError(f"Checkpoint does not contain model_config: {checkpoint_path}")

    model_cfg = ModelConfig(**state["model_config"])
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match checkpoint model vocab size "
            f"({model_cfg.vocab_size})."
        )
    verify_checkpoint_fingerprints(
        state,
        model_config=model_cfg,
        tokenizer_model_path=tokenizer_path,
        context="chat checkpoint",
    )

    model = GPT(model_cfg)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, model_cfg, device
