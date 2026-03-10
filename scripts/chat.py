from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, load_chat_config
from src.generation import build_chat_prompt_tokens, generate
from src.model import GPT
from src.tokenizer_utils import SentencePieceTokenizer
from src.utils import assert_exists, get_device, load_torch_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive local chat CLI.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/chat.yaml",
        help="Path to chat YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path override.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt override.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature override.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k override.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional generation length override.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Enable lightweight JSON-style prompting.",
    )
    return parser.parse_args()


def strip_hidden_stop_tokens(tokenizer: SentencePieceTokenizer, token_ids: list[int]) -> list[int]:
    hidden = {
        tokenizer.eos_id,
        tokenizer.pad_id,
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
    }
    out = list(token_ids)
    while out and out[-1] in hidden:
        out.pop()
    return out


def main() -> None:
    args = parse_args()
    cfg = load_chat_config(args.config)

    checkpoint_path = args.checkpoint or cfg.checkpoint_path
    tokenizer_path = cfg.tokenizer_model_path
    system_prompt = args.system_prompt or cfg.system_prompt
    temperature = cfg.temperature if args.temperature is None else args.temperature
    top_k = cfg.top_k if args.top_k is None else args.top_k
    max_new_tokens = cfg.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    json_mode = cfg.json_mode or args.json_mode

    checkpoint_path = assert_exists(checkpoint_path, "Chat checkpoint")
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    device = get_device(cfg.device)

    state = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if "model_config" not in state:
        raise ValueError(f"Checkpoint does not contain model_config: {checkpoint_path}")

    model_cfg = ModelConfig(**state["model_config"])
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match checkpoint model vocab size "
            f"({model_cfg.vocab_size})."
        )

    model = GPT(model_cfg)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    print("Tiny LLM chat")
    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"json_mode: {json_mode}")
    print("type 'exit', 'quit', or ':q' to leave")
    print("type 'clear' to reset chat history")
    print()

    history: list[tuple[str, str]] = []

    stop_ids = [
        tokenizer.eos_id,
        tokenizer.token_to_id("<|system|>"),
        tokenizer.token_to_id("<|user|>"),
        tokenizer.token_to_id("<|assistant|>"),
    ]
    if json_mode:
        stop_ids.append(tokenizer.token_to_id("</json>"))

    while True:
        try:
            user_text = input("user> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_text:
            continue

        lowered = user_text.lower()
        if lowered in {"exit", "quit", ":q"}:
            print("Exiting.")
            break
        if lowered == "clear":
            history.clear()
            print("History cleared.")
            continue

        prompt_tokens = build_chat_prompt_tokens(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            history=history,
            user_message=user_text,
            block_size=model_cfg.block_size,
            max_history_turns=cfg.max_history_turns,
            json_mode=json_mode,
        )

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            stop_token_ids=stop_ids,
        )

        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        generated_ids = strip_hidden_stop_tokens(tokenizer, generated_ids)
        response = tokenizer.decode(generated_ids).strip()

        if not response:
            response = "(empty response)"

        print(f"assistant> {response}")
        print()

        history.append((user_text, response))


if __name__ == "__main__":
    main()