from __future__ import annotations

import json

import pytest

from src.capabilities import (
    AUTO_THINK_TOKEN,
    END_THINK_TOKEN,
    END_TOOL_CALL_TOKEN,
    FINAL_TOKEN,
    NO_THINK_TOKEN,
    THINK_TOKEN,
    TOOL_CALL_TOKEN,
    ThinkingMode,
    build_capability_prefix,
    format_tool_result,
    parse_assistant_response,
    parse_tool_call,
)


def test_capability_prefix_selects_thinking_mode_and_serializes_tools() -> None:
    prefix = build_capability_prefix(
        ThinkingMode.AUTO,
        [
            {
                "name": "calculator",
                "description": "Evaluate arithmetic",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                },
            }
        ],
    )
    assert prefix.startswith(AUTO_THINK_TOKEN)
    assert '"name":"calculator"' in prefix


def test_thinking_off_uses_control_token() -> None:
    assert build_capability_prefix("off").startswith(NO_THINK_TOKEN)


def test_parse_reasoning_tool_call_and_final() -> None:
    payload = {"name": "calculator", "arguments": {"expression": "2+2"}, "id": "c1"}
    text = (
        f"{THINK_TOKEN}\nNeed arithmetic.\n{END_THINK_TOKEN}\n"
        f"{TOOL_CALL_TOKEN}\n{json.dumps(payload)}\n{END_TOOL_CALL_TOKEN}\n"
        f"{FINAL_TOKEN}\nThe answer is 4."
    )
    parsed = parse_assistant_response(text)
    assert parsed.thinking == "Need arithmetic."
    assert parsed.final == "The answer is 4."
    assert parsed.tool_calls[0].name == "calculator"
    assert parsed.tool_calls[0].arguments == {"expression": "2+2"}


def test_parse_without_final_token_does_not_leak_hidden_sections() -> None:
    text = (
        f"{THINK_TOKEN}\nPrivate reasoning.\n{END_THINK_TOKEN}\n"
        "Visible answer."
    )
    parsed = parse_assistant_response(text)
    assert parsed.thinking == "Private reasoning."
    assert parsed.final == "Visible answer."
    assert THINK_TOKEN not in parsed.final
    assert "Private reasoning" not in parsed.final


def test_parse_tool_call_rejects_non_object_arguments() -> None:
    with pytest.raises(ValueError, match="arguments"):
        parse_tool_call('{"name":"x","arguments":[]}')


def test_format_tool_result_is_machine_readable() -> None:
    result = format_tool_result("calculator", {"value": 4}, "c1")
    assert '"name":"calculator"' in result
    assert '"id":"c1"' in result
    assert '"value":4' in result
