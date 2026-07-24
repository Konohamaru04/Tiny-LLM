from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset


DEFAULT_SYSTEM = "You are a helpful assistant that can reason, plan, and use tools."


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def bounded(stream: Iterable[dict[str, Any]], limit: int) -> Iterator[dict[str, Any]]:
    for index, row in enumerate(stream):
        if index >= limit:
            break
        yield row


def prepare_fineweb(output_dir: Path, limit: int) -> int:
    """Stream a small educational pretraining shard as Markdown documents."""
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, row in enumerate(bounded(dataset, limit)):
        text = str(row.get("text", "")).strip()
        if len(text) < 200:
            continue
        (output_dir / f"fineweb_{index:07d}.md").write_text(text, encoding="utf-8")
        count += 1
    return count


def extract_thinking_and_final(solution: str) -> tuple[str, str]:
    solution = solution.strip()
    match = re.search(r"<think>(.*?)</think>(.*)", solution, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return solution, ""


def prepare_reasoning(limit: int) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        "open-r1/OpenR1-Math-220k",
        name="default",
        split="train",
        streaming=True,
    )
    emitted = 0
    for row in dataset:
        problem = str(row.get("problem", "")).strip()
        solution = str(row.get("solution", "")).strip()
        if not solution:
            messages = row.get("messages") or []
            if isinstance(messages, list):
                assistant = next(
                    (
                        str(item.get("content", ""))
                        for item in reversed(messages)
                        if isinstance(item, dict) and item.get("role") == "assistant"
                    ),
                    "",
                )
                user = next(
                    (
                        str(item.get("content", ""))
                        for item in messages
                        if isinstance(item, dict) and item.get("role") == "user"
                    ),
                    "",
                )
                problem = problem or user
                solution = assistant
        if not problem or not solution:
            continue
        thinking, final = extract_thinking_and_final(solution)
        if not final:
            final = str(row.get("answer", "")).strip() or solution
        yield {
            "system": DEFAULT_SYSTEM,
            "user": problem,
            "thinking_mode": "on",
            "tools": [],
            "thinking": thinking,
            "final": final,
        }
        emitted += 1
        if emitted >= limit:
            return


def role_name(message: dict[str, Any]) -> str:
    return str(message.get("role", message.get("from", ""))).lower()


def message_content(message: dict[str, Any]) -> str:
    return str(message.get("content", message.get("value", ""))).strip()


def normalize_tool_definition(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = value.get("tools", value.get("functions", [value]))
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function", item)
        if not isinstance(function, dict) or not function.get("name"):
            continue
        output.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        )
    return output


def normalize_tool_call(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    calls = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function", item)
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        if name and isinstance(arguments, dict):
            calls.append({"name": name, "arguments": arguments, "id": item.get("id", "")})
    return calls


def prepare_tool_calling(limit: int) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        "Johin/function-calling-dataset",
        split="train",
        streaming=True,
    )
    emitted = 0
    for row in dataset:
        messages = row.get("messages") or row.get("conversations") or []
        if not isinstance(messages, list):
            continue
        system = DEFAULT_SYSTEM
        user = ""
        final = ""
        calls: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = role_name(message)
            content = message_content(message)
            if role in {"system"} and content:
                system = content
            elif role in {"user", "human"} and not user:
                user = content
            elif role in {"assistant", "gpt"}:
                calls.extend(
                    normalize_tool_call(
                        message.get("tool_calls", message.get("function_call"))
                    )
                )
                if content:
                    final = content
        metadata = row.get("metadata", {})
        tools = normalize_tool_definition(
            row.get("tools", metadata.get("functions", metadata.get("tools", [])))
            if isinstance(metadata, dict)
            else row.get("tools", [])
        )
        if not user or (not calls and not final):
            continue
        yield {
            "system": system,
            "user": user,
            "thinking_mode": "auto",
            "tools": tools,
            "thinking": "",
            "tool_calls": calls,
            "final": final,
        }
        emitted += 1
        if emitted >= limit:
            return


def direct_examples(limit: int) -> Iterator[dict[str, Any]]:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    emitted = 0
    for row in dataset:
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        user = next((message_content(m) for m in messages if role_name(m) == "user"), "")
        assistant = next(
            (message_content(m) for m in reversed(messages) if role_name(m) == "assistant"),
            "",
        )
        if not user or not assistant:
            continue
        yield {
            "system": DEFAULT_SYSTEM,
            "user": user,
            "thinking_mode": "off",
            "tools": [],
            "final": assistant,
        }
        emitted += 1
        if emitted >= limit:
            return


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_size = max(1, int(len(rows) * val_fraction))
    return rows[val_size:], rows[:val_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public data for Tiny-LLM unified MoE training")
    parser.add_argument("--fineweb-docs", type=int, default=20000)
    parser.add_argument("--direct", type=int, default=20000)
    parser.add_argument("--reasoning", type=int, default=20000)
    parser.add_argument("--tools", type=int, default=10000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_count = prepare_fineweb(Path("data/raw/public"), args.fineweb_docs)
    rows = list(direct_examples(args.direct))
    rows.extend(prepare_reasoning(args.reasoning))
    rows.extend(prepare_tool_calling(args.tools))
    train_rows, val_rows = split_rows(rows, args.val_fraction, args.seed)
    train_count = write_jsonl(Path("data/sft/unified_train.jsonl"), train_rows)
    val_count = write_jsonl(Path("data/sft/unified_val.jsonl"), val_rows)
    print(f"Prepared {raw_count} pretraining docs, {train_count} SFT train rows, {val_count} validation rows")


if __name__ == "__main__":
    main()
