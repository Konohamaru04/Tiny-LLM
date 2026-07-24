from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat_runtime import (
    generate_chat_response,
    load_chat_runtime,
    load_personas,
    load_session,
    resolve_persona,
    run_agent_turn,
    save_session,
)
from src.config import load_chat_config
from src.tools import build_default_tool_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive local chat CLI.")
    parser.add_argument("--config", type=str, default="configs/chat.yaml", help="Path to chat YAML config.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override.")
    parser.add_argument("--system-prompt", type=str, default="", help="Optional system prompt override.")
    parser.add_argument("--persona", type=str, default="", help="Optional persona name override.")
    parser.add_argument("--personas-path", type=str, default="", help="Optional personas JSON path override.")
    parser.add_argument("--session-file", type=str, default="", help="Optional chat session JSON path.")
    parser.add_argument("--list-personas", action="store_true", help="List available personas and exit.")
    parser.add_argument("--temperature", type=float, default=None, help="Optional temperature override.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k override.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional generation length override.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional repetition penalty override (>= 1.0).")
    parser.add_argument("--json-mode", action="store_true", help="Enable lightweight JSON-style prompting.")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    return parser.parse_args()


def print_personas(personas_path: str) -> None:
    personas = load_personas(personas_path)
    print("Available personas:")
    for name in sorted(personas):
        persona = personas[name]
        suffix = f" - {persona.description}" if persona.description else ""
        print(f"  {name}{suffix}")


def print_help() -> None:
    print("Commands:")
    print("  :help                 show commands")
    print("  :clear                clear conversation history")
    print("  :history              print current conversation history")
    print("  :save                 save the current session file")
    print("  :persona NAME         switch to a persona from the personas file")
    print("  :json on|off          toggle JSON prompting")
    print("  :system TEXT          replace the active system prompt")
    print("  :session              show the current session file")


def main() -> None:
    args = parse_args()
    cfg = load_chat_config(args.config)

    personas_path = args.personas_path or cfg.personas_path
    personas = load_personas(personas_path)
    if args.list_personas:
        print_personas(personas_path)
        return

    session_file = args.session_file or cfg.session_file
    persona_name = args.persona or cfg.default_persona
    json_mode = cfg.json_mode or args.json_mode
    system_prompt_override = args.system_prompt.strip()
    temperature = cfg.temperature if args.temperature is None else args.temperature
    top_k = cfg.top_k if args.top_k is None else args.top_k
    max_new_tokens = cfg.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens
    repetition_penalty = cfg.repetition_penalty if args.repetition_penalty is None else args.repetition_penalty
    stream = cfg.stream and not args.no_stream

    session = load_session(session_file) if session_file else {"history": []}
    history = list(session.get("history", []))
    if not system_prompt_override:
        session_persona = str(session.get("persona", "")).strip()
        if session_persona:
            persona_name = session_persona

    active_persona = resolve_persona(
        personas,
        persona_name,
        fallback_system_prompt=system_prompt_override or cfg.system_prompt,
        fallback_json_mode=json_mode,
    )
    system_prompt = system_prompt_override or str(session.get("system_prompt", "")).strip() or active_persona.system_prompt
    json_mode = bool(session.get("json_mode", json_mode or active_persona.json_mode))

    model, tokenizer, model_cfg, device = load_chat_runtime(
        cfg,
        checkpoint_override=args.checkpoint,
    )

    print("Tiny LLM chat")
    print(f"device: {device}")
    print(f"persona: {active_persona.name}")
    print(f"json_mode: {json_mode}")
    print(f"thinking_mode: {cfg.thinking_mode}")
    print(f"tool_mode: {cfg.tool_mode}")
    if session_file:
        print(f"session_file: {Path(session_file).resolve()}")
    print("type ':help' for commands")
    print()

    def persist_session() -> None:
        if not session_file:
            return
        save_session(
            session_file,
            history=history,
            persona_name=active_persona.name,
            system_prompt=system_prompt,
            json_mode=json_mode,
        )

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
            persist_session()
            print("Exiting.")
            break

        if lowered.startswith(":"):
            if lowered == ":help":
                print_help()
            elif lowered == ":clear":
                history.clear()
                persist_session()
                print("History cleared.")
            elif lowered == ":history":
                if not history:
                    print("(history is empty)")
                for idx, (past_user, past_assistant) in enumerate(history, start=1):
                    print(f"{idx}. user: {past_user}")
                    print(f"   assistant: {past_assistant}")
            elif lowered == ":save":
                if not session_file:
                    print("No session file configured.")
                else:
                    persist_session()
                    print(f"Session saved to {Path(session_file).resolve()}")
            elif lowered.startswith(":persona "):
                requested = user_text.split(" ", 1)[1].strip()
                active_persona = resolve_persona(
                    personas,
                    requested,
                    fallback_system_prompt=system_prompt,
                    fallback_json_mode=json_mode,
                )
                system_prompt = active_persona.system_prompt
                json_mode = active_persona.json_mode
                persist_session()
                print(f"Switched to persona: {active_persona.name}")
            elif lowered.startswith(":json "):
                option = user_text.split(" ", 1)[1].strip().lower()
                if option not in {"on", "off"}:
                    print("Usage: :json on|off")
                else:
                    json_mode = option == "on"
                    persist_session()
                    print(f"json_mode: {json_mode}")
            elif lowered.startswith(":system "):
                system_prompt = user_text.split(" ", 1)[1].strip()
                persist_session()
                print("Updated system prompt.")
            elif lowered == ":session":
                if session_file:
                    print(Path(session_file).resolve())
                else:
                    print("No session file configured.")
            else:
                print("Unknown command. Type ':help' for commands.")
            print()
            continue

        print("assistant> ", end="", flush=True)

        def on_chunk(chunk: str) -> None:
            print(chunk, end="", flush=True)

        if cfg.tool_mode:
            agent_result = run_agent_turn(
                model=model,
                tokenizer=tokenizer,
                model_cfg=model_cfg,
                device=device,
                system_prompt=system_prompt,
                history=history,
                user_message=user_text,
                max_history_turns=cfg.max_history_turns,
                registry=build_default_tool_registry(),
                max_tool_rounds=cfg.max_tool_rounds,
                json_mode=json_mode,
                thinking_mode=cfg.thinking_mode,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
            response = agent_result.response
            for event in agent_result.tool_events:
                print(
                    f"\n[tool] {event.call.name}"
                    f"({json.dumps(event.call.arguments, ensure_ascii=False)}) -> "
                    f"{json.dumps(event.output, ensure_ascii=False)}"
                )
            if agent_result.reached_tool_limit:
                print("\n[tool] maximum tool rounds reached")
            print(response, end="", flush=True)
        else:
            response = generate_chat_response(
                model=model,
                tokenizer=tokenizer,
                model_cfg=model_cfg,
                device=device,
                system_prompt=system_prompt,
                history=history,
                user_message=user_text,
                max_history_turns=cfg.max_history_turns,
                json_mode=json_mode,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                on_text_chunk=on_chunk if stream else None,
                thinking_mode=cfg.thinking_mode,
            )

        if not stream and not cfg.tool_mode:
            print(response, end="", flush=True)
        if not response:
            response = "(empty response)"
            if stream:
                print(response, end="", flush=True)
        print("\n")

        history.append((user_text, response))
        persist_session()


if __name__ == "__main__":
    main()
