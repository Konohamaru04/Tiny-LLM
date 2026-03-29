from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from src.generation import build_chat_prompt_tokens, generate
from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import assert_exists, resolve_path


def load_eval_prompts(path: str | Path) -> list[dict[str, Any]]:
    prompts_path = assert_exists(path, "Evaluation prompts")
    prompts: list[dict[str, Any]] = []

    def normalize_history(raw_history: Any, line_no: int | None = None) -> list[tuple[str, str]]:
        if raw_history in (None, []):
            return []
        if not isinstance(raw_history, list):
            raise ValueError(f"Prompt history must be a list at {prompts_path}:{line_no or 1}")

        normalized: list[tuple[str, str]] = []
        for item in raw_history:
            if not isinstance(item, dict):
                raise ValueError(f"History items must be objects at {prompts_path}:{line_no or 1}")
            user_text = str(item.get("user", "")).strip()
            assistant_text = str(item.get("assistant", "")).strip()
            if not user_text and not assistant_text:
                continue
            normalized.append((user_text, assistant_text))
        return normalized

    if prompts_path.suffix.lower() == ".json":
        with prompts_path.open("r", encoding="utf-8") as f:
            raw_items = json.load(f)
        if not isinstance(raw_items, list):
            raise ValueError(f"Prompt JSON must contain a list: {prompts_path}")
        iterable = list(enumerate(raw_items, start=1))
    else:
        raw_rows: list[tuple[int, Any]] = []
        with prompts_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                raw_rows.append((line_no, json.loads(raw)))
        iterable = raw_rows

    for idx, raw_item in iterable:
        if not isinstance(raw_item, dict):
            raise ValueError(f"Prompt row must be an object at {prompts_path}:{idx}")
        user_text = str(raw_item.get("user", "")).strip()
        if not user_text:
            raise ValueError(f"Prompt row is missing a non-empty `user` field at {prompts_path}:{idx}")

        prompts.append(
            {
                "name": str(raw_item.get("name") or f"prompt_{idx:03d}"),
                "system": str(raw_item.get("system", "")).strip(),
                "user": user_text,
                "history": normalize_history(raw_item.get("history"), idx),
                "json_mode": bool(raw_item.get("json_mode", False)),
            }
        )

    if not prompts:
        raise ValueError(f"No prompts were loaded from: {prompts_path}")
    return prompts


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: Any):
    if not amp_enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


@torch.no_grad()
def evaluate_loss_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    amp_enabled: bool = False,
    amp_dtype: Any = None,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()

    losses: list[float] = []
    batch_count = 0

    with _autocast_context(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = batch
            x = x.to(device, non_blocking=device.type == "cuda")
            y = y.to(device, non_blocking=device.type == "cuda")
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model returned no loss during evaluation.")
            losses.append(float(loss.detach().cpu().item()))
            batch_count += 1

    if was_training:
        model.train()

    if not losses:
        raise RuntimeError("Evaluation loader produced zero batches.")

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(min(avg_loss, 20.0))
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "batches_evaluated": batch_count,
    }


def _strip_hidden_stop_tokens(tokenizer: SentencePieceTokenizer, token_ids: Sequence[int]) -> list[int]:
    hidden = {
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
    }
    out = list(int(token) for token in token_ids)
    while out and out[-1] in hidden:
        out.pop()
    return out


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if "<|json|>" in stripped and "</json>" in stripped:
        start = stripped.find("<|json|>") + len("<|json|>")
        end = stripped.rfind("</json>")
        return stripped[start:end].strip()
    return stripped


def is_valid_json_response(text: str) -> bool:
    candidate = _extract_json_payload(text)
    if not candidate or candidate[0] not in "[{":
        return False
    try:
        json.loads(candidate)
    except Exception:
        return False
    return True


def repetition_score(text: str, ngram_size: int = 3) -> float:
    words = text.strip().split()
    if len(words) < ngram_size:
        return 0.0
    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


@torch.inference_mode()
def run_generation_eval(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    prompts: Sequence[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float = 1.0,
    default_system_prompt: str = "",
    max_history_turns: int = 4,
    default_json_mode: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")

    was_training = model.training
    model.eval()

    samples: list[dict[str, Any]] = []
    empty_count = 0
    json_valid_count = 0
    json_requested_count = 0
    repetition_values: list[float] = []

    stop_ids_base = [
        tokenizer.eos_id,
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
    ]

    for prompt in prompts:
        json_mode = bool(prompt.get("json_mode", default_json_mode) or default_json_mode)
        system_prompt = str(prompt.get("system") or default_system_prompt).strip()
        history = prompt.get("history") or []

        prompt_tokens = build_chat_prompt_tokens(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            history=history,
            user_message=str(prompt["user"]),
            block_size=model.config.block_size,
            max_history_turns=max_history_turns,
            json_mode=json_mode,
        )
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        stop_ids = list(stop_ids_base)
        if json_mode:
            stop_ids.append(tokenizer.token_to_id("</json>"))

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
        generated_ids = _strip_hidden_stop_tokens(tokenizer, generated_ids)
        response = tokenizer.decode(generated_ids).strip()

        response_is_empty = not response
        response_repetition = repetition_score(response)
        response_is_valid_json = is_valid_json_response(response)

        if response_is_empty:
            empty_count += 1
        if json_mode:
            json_requested_count += 1
            if response_is_valid_json:
                json_valid_count += 1
        repetition_values.append(response_repetition)

        samples.append(
            {
                "name": str(prompt.get("name", "prompt")),
                "system": system_prompt,
                "user": str(prompt["user"]),
                "history": history,
                "json_mode": json_mode,
                "response": response,
                "response_chars": len(response),
                "response_words": len(response.split()),
                "response_is_empty": response_is_empty,
                "response_is_valid_json": response_is_valid_json,
                "response_repetition_3gram": round(response_repetition, 4),
            }
        )

    if was_training:
        model.train()

    avg_chars = sum(sample["response_chars"] for sample in samples) / len(samples)
    avg_words = sum(sample["response_words"] for sample in samples) / len(samples)
    avg_repetition = sum(repetition_values) / len(repetition_values) if repetition_values else 0.0

    summary = {
        "num_prompts": len(samples),
        "avg_response_chars": round(avg_chars, 2),
        "avg_response_words": round(avg_words, 2),
        "empty_response_rate": round(empty_count / len(samples), 4),
        "avg_repetition_3gram": round(avg_repetition, 4),
        "json_prompt_count": json_requested_count,
        "json_valid_rate": round((json_valid_count / json_requested_count), 4) if json_requested_count else 0.0,
    }
    return summary, samples
