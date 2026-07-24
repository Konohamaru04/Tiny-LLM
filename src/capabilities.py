from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class ThinkingMode(str, Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


THINK_TOKEN = "<|think|>"
END_THINK_TOKEN = "<|end_think|>"
NO_THINK_TOKEN = "<|no_think|>"
AUTO_THINK_TOKEN = "<|auto_think|>"
TOOLS_TOKEN = "<|tools|>"
END_TOOLS_TOKEN = "<|end_tools|>"
TOOL_CALL_TOKEN = "<|tool_call|>"
END_TOOL_CALL_TOKEN = "<|end_tool_call|>"
TOOL_RESULT_TOKEN = "<|tool_result|>"
END_TOOL_RESULT_TOKEN = "<|end_tool_result|>"
FINAL_TOKEN = "<|final|>"

CAPABILITY_TOKENS = (
    THINK_TOKEN,
    END_THINK_TOKEN,
    NO_THINK_TOKEN,
    AUTO_THINK_TOKEN,
    TOOLS_TOKEN,
    END_TOOLS_TOKEN,
    TOOL_CALL_TOKEN,
    END_TOOL_CALL_TOKEN,
    TOOL_RESULT_TOKEN,
    END_TOOL_RESULT_TOKEN,
    FINAL_TOKEN,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str = ""


@dataclass(frozen=True)
class ParsedAssistantResponse:
    thinking: str
    final: str
    tool_calls: tuple[ToolCall, ...]
    raw_text: str


def normalize_thinking_mode(mode: str | ThinkingMode) -> ThinkingMode:
    if isinstance(mode, ThinkingMode):
        return mode
    try:
        return ThinkingMode(str(mode).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ThinkingMode)
        raise ValueError(f"Unsupported thinking mode `{mode}`. Expected one of: {allowed}") from exc


def thinking_control_token(mode: str | ThinkingMode) -> str:
    normalized = normalize_thinking_mode(mode)
    if normalized is ThinkingMode.OFF:
        return NO_THINK_TOKEN
    if normalized is ThinkingMode.ON:
        return THINK_TOKEN
    return AUTO_THINK_TOKEN


def serialize_tools(tools: Sequence[ToolDefinition | Mapping[str, Any]]) -> str:
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        payload = tool.as_dict() if isinstance(tool, ToolDefinition) else dict(tool)
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Every tool definition must include a non-empty `name`.")
        payload.setdefault("description", "")
        payload.setdefault("parameters", {"type": "object", "properties": {}})
        normalized.append(payload)
    if not normalized:
        return ""
    body = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOLS_TOKEN}\n{body}\n{END_TOOLS_TOKEN}\n"


def build_capability_prefix(
    thinking_mode: str | ThinkingMode,
    tools: Sequence[ToolDefinition | Mapping[str, Any]] = (),
) -> str:
    return f"{thinking_control_token(thinking_mode)}\n{serialize_tools(tools)}"


def _extract_sections(text: str, start: str, end: str) -> list[str]:
    sections: list[str] = []
    cursor = 0
    while True:
        start_index = text.find(start, cursor)
        if start_index < 0:
            break
        content_start = start_index + len(start)
        end_index = text.find(end, content_start)
        if end_index < 0:
            break
        sections.append(text[content_start:end_index].strip())
        cursor = end_index + len(end)
    return sections


def parse_tool_call(payload: str) -> ToolCall:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Tool call payload must be a JSON object.")
    name = str(value.get("name", "")).strip()
    arguments = value.get("arguments", {})
    call_id = str(value.get("id", "")).strip()
    if not name:
        raise ValueError("Tool call payload must include a non-empty `name`.")
    if not isinstance(arguments, dict):
        raise ValueError("Tool call `arguments` must be a JSON object.")
    return ToolCall(name=name, arguments=arguments, call_id=call_id)


def parse_assistant_response(text: str) -> ParsedAssistantResponse:
    raw = text.strip()
    thinking_sections = _extract_sections(raw, THINK_TOKEN, END_THINK_TOKEN)
    tool_sections = _extract_sections(raw, TOOL_CALL_TOKEN, END_TOOL_CALL_TOKEN)
    tool_calls = tuple(parse_tool_call(section) for section in tool_sections)

    final_index = raw.rfind(FINAL_TOKEN)
    if final_index >= 0:
        final = raw[final_index + len(FINAL_TOKEN) :].strip()
    else:
        final = raw
        for start, end in (
            (THINK_TOKEN, END_THINK_TOKEN),
            (TOOL_CALL_TOKEN, END_TOOL_CALL_TOKEN),
        ):
            for section in _extract_sections(final, start, end):
                final = final.replace(f"{start}{section}{end}", "")
        final = final.strip()

    return ParsedAssistantResponse(
        thinking="\n\n".join(thinking_sections),
        final=final,
        tool_calls=tool_calls,
        raw_text=raw,
    )


def format_tool_result(name: str, result: Any, call_id: str = "") -> str:
    payload = {"name": name, "result": result}
    if call_id:
        payload["id"] = call_id
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_RESULT_TOKEN}\n{encoded}\n{END_TOOL_RESULT_TOKEN}"


def validate_tool_calls(calls: Iterable[ToolCall], allowed_names: Iterable[str]) -> None:
    allowed = set(allowed_names)
    unknown = sorted({call.name for call in calls if call.name not in allowed})
    if unknown:
        raise ValueError(f"Model requested unknown tools: {', '.join(unknown)}")
