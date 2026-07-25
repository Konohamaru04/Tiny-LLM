from __future__ import annotations

import unittest

from src.tools import (
    ParsedToolCall,
    build_default_tool_registry,
    parse_assistant_response,
    parse_tool_calls,
    split_thinking_block,
)


class ToolRuntimeTests(unittest.TestCase):
    def test_response_parser_hides_reasoning_and_returns_final_answer(self) -> None:
        parsed = parse_assistant_response(
            "<|think|>Private scratch work.</|think|>\n"
            "<|final|>\nThe visible answer."
        )

        self.assertEqual(parsed.reasoning, "Private scratch work.")
        self.assertEqual(parsed.final, "The visible answer.")
        self.assertNotIn("Private scratch", parsed.final)

    def test_response_parser_drops_unclosed_reasoning(self) -> None:
        parsed = parse_assistant_response("<|think|>unfinished hidden reasoning")

        self.assertEqual(parsed.final, "")

    def test_tool_markup_inside_reasoning_is_not_executed(self) -> None:
        parsed = parse_assistant_response(
            "<|think|>"
            '<|tool_call|>{"name":"calculator","arguments":{"expression":"1"}}'
            "</|tool_call|>"
            "</|think|>"
            "<|final|>No tool is needed."
        )

        self.assertEqual(parsed.final, "No tool is needed.")
        self.assertEqual(parsed.tool_calls, ())

    def test_split_thinking_block_preserves_visible_answer(self) -> None:
        reasoning, content = split_thinking_block(
            "<|think|>Need the calculator.</|think|>\nThe result is ready."
        )

        self.assertEqual(reasoning, "Need the calculator.")
        self.assertEqual(content, "The result is ready.")

    def test_parse_and_execute_calculator_call(self) -> None:
        calls = parse_tool_calls(
            '<|tool_call|>{"id":"math_1","name":"calculator",'
            '"arguments":{"expression":"sqrt(81) + 3"}}</|tool_call|>'
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].id, "math_1")
        output = build_default_tool_registry().execute(calls[0])
        self.assertTrue(output["ok"])
        self.assertEqual(output["result"]["value"], 12.0)

    def test_calculator_rejects_code_execution(self) -> None:
        call = ParsedToolCall(
            id="bad_1",
            name="calculator",
            arguments={"expression": "__import__('os').system('echo unsafe')"},
        )

        output = build_default_tool_registry().execute(call)

        self.assertFalse(output["ok"])
        self.assertIn("Unsupported", output["error"])
