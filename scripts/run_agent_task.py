from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat_runtime import load_chat_runtime
from src.config import load_chat_config
from src.long_horizon import (
    build_runtime_fingerprint,
    run_long_horizon_task,
)
from src.tools import build_default_tool_registry
from src.utils import resolve_path, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume a checkpointed long-horizon tool task. State is "
            "saved atomically after every model step."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/chat_long_horizon.yaml",
        help="Long-horizon chat YAML.",
    )
    parser.add_argument(
        "--state",
        required=True,
        help="Task state JSON. Reusing it resumes the task.",
    )
    parser.add_argument(
        "--objective",
        default="",
        help="Required only when creating a new task state.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional .safetensors checkpoint override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=64,
        help="Persistent total model-step budget; increase it to extend a task.",
    )
    parser.add_argument(
        "--steps-per-run",
        type=int,
        default=0,
        help="Pause after this many model steps; 0 uses the remaining budget.",
    )
    parser.add_argument(
        "--max-wall-time-seconds",
        type=float,
        default=0.0,
        help="Checkpoint and pause after this run time; 0 disables the limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_chat_config(args.config)
    registry = build_default_tool_registry()
    checkpoint_path = args.checkpoint or cfg.checkpoint_path
    model, tokenizer, model_cfg, device = load_chat_runtime(
        cfg,
        checkpoint_override=args.checkpoint,
    )
    fingerprint = build_runtime_fingerprint(
        checkpoint_sha256=sha256_file(checkpoint_path),
        tokenizer_sha256=sha256_file(cfg.tokenizer_model_path),
        model_config=model_cfg,
        tool_schemas=registry.schemas(),
    )
    state = run_long_horizon_task(
        state_path=args.state,
        objective=args.objective,
        system_prompt=cfg.system_prompt,
        max_steps=args.max_steps,
        runtime_fingerprint=fingerprint,
        model=model,
        tokenizer=tokenizer,
        model_cfg=model_cfg,
        device=device,
        registry=registry,
        json_mode=cfg.json_mode,
        thinking_mode=cfg.thinking_mode,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        max_new_tokens=cfg.max_new_tokens,
        repetition_penalty=cfg.repetition_penalty,
        steps_per_run=args.steps_per_run or None,
        max_wall_time_seconds=args.max_wall_time_seconds or None,
    )
    summary = {
        "task_id": state.task_id,
        "status": state.status,
        "steps_completed": state.steps_completed,
        "max_steps": state.max_steps,
        "tool_events": len(state.events),
        "state_path": str(resolve_path(args.state)),
        "final_response": state.final_response,
        "last_error": state.last_error,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
