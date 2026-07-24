from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.chat_format import normalize_tool_call


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def as_message_call(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    self.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if not definition.name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", definition.name):
            raise ValueError(f"Invalid tool name: {definition.name!r}")
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]

    def execute(self, call: ParsedToolCall) -> dict[str, Any]:
        definition = self._tools.get(call.name)
        if definition is None:
            return {"ok": False, "error": f"Unknown tool: {call.name}"}
        try:
            result = definition.handler(**call.arguments)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MATH_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "round": round,
}
_MATH_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _evaluate_math_node(node: ast.AST, depth: int = 0) -> float | int:
    if depth > 24:
        raise ValueError("Expression is too deeply nested")
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _MATH_CONSTANTS:
        return _MATH_CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_math_node(node.left, depth + 1)
        right = _evaluate_math_node(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
            raise ValueError("Exponent magnitude must be <= 100")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_math_node(node.operand, depth + 1))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = _MATH_FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError(f"Unsupported function: {node.func.id}")
        args = [_evaluate_math_node(arg, depth + 1) for arg in node.args]
        return function(*args)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> dict[str, Any]:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expression) > 256:
        raise ValueError("expression must be at most 256 characters")
    tree = ast.parse(expression, mode="eval")
    result = _evaluate_math_node(tree.body)
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("result is not finite")
    return {"expression": expression, "value": result}


def current_datetime(timezone_name: str = "UTC") -> dict[str, str]:
    if timezone_name.upper() == "UTC":
        tz = timezone.utc
    else:
        try:
            tz = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unknown IANA timezone: {timezone_name}") from exc
    now = datetime.now(tz)
    return {
        "timezone": timezone_name,
        "iso": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.time().isoformat(timespec="seconds"),
    }


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calculator",
            description="Safely evaluate a numeric arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression using numbers and basic math functions.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=calculate,
        )
    )
    registry.register(
        ToolDefinition(
            name="current_datetime",
            description="Return the current date and time in an IANA timezone.",
            parameters={
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "IANA timezone such as UTC or Asia/Kolkata.",
                        "default": "UTC",
                    }
                },
                "additionalProperties": False,
            },
            handler=current_datetime,
        )
    )
    return registry


_TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call\|>\s*(.*?)\s*</\|tool_call\|>",
    flags=re.DOTALL,
)
_THINK_PATTERN = re.compile(
    r"<\|think\|>\s*(.*?)\s*</\|think\|>",
    flags=re.DOTALL,
)


def parse_tool_calls(text: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for index, match in enumerate(_TOOL_CALL_PATTERN.finditer(text)):
        try:
            raw = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, Mapping):
            continue
        canonical = normalize_tool_call(raw, index)
        if not canonical["name"]:
            continue
        calls.append(
            ParsedToolCall(
                id=canonical["id"],
                name=canonical["name"],
                arguments=canonical["arguments"],
            )
        )
    return calls


def strip_tool_call_blocks(text: str) -> str:
    return _TOOL_CALL_PATTERN.sub("", text).strip()


def split_thinking_block(text: str) -> tuple[str, str]:
    """Return the first explicit reasoning block and the remaining visible content."""
    match = _THINK_PATTERN.search(text)
    if match is None:
        return "", text.strip()
    reasoning = match.group(1).strip()
    content = (text[: match.start()] + text[match.end() :]).strip()
    return reasoning, content
