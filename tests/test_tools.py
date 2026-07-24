from __future__ import annotations

import unittest

from src.tools import (
    ParsedToolCall,
    build_default_tool_registry,
    parse_tool_calls,
    split_thinking_block,
)


class ToolRuntimeTests(unittest.TestCase):
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
