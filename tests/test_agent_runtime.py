from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.agent_runtime import AgentState, LongHorizonAgent
from src.capabilities import ParsedAssistantResponse, ToolCall, ToolDefinition
from src.config import ModelConfig


class ScriptedAgent(LongHorizonAgent):
    def __init__(self, responses: list[ParsedAssistantResponse], **kwargs):
        self.responses = list(responses)
        self.prompts: list[str] = []
        super().__init__(**kwargs)

    def _run_model(self, user_message: str) -> ParsedAssistantResponse:
        self.prompts.append(user_message)
        if not self.responses:
            raise AssertionError("Scripted agent ran out of responses")
        return self.responses.pop(0)


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="calculator",
        description="Evaluate a tiny arithmetic request",
        parameters={"type": "object", "properties": {"value": {"type": "integer"}}},
    )


def test_long_horizon_agent_executes_tool_and_feeds_protocol_result(tmp_path: Path) -> None:
    responses = [
        ParsedAssistantResponse(
            thinking="Need the calculator.",
            final="",
            tool_calls=(ToolCall(name="calculator", arguments={"value": 4}, call_id="c1"),),
            raw_text='<|tool_call|>{"name":"calculator","arguments":{"value":4},"id":"c1"}<|end_tool_call|>',
        ),
        ParsedAssistantResponse(
            thinking="",
            final="The result is 8.",
            tool_calls=(),
            raw_text="<|final|>The result is 8.",
        ),
    ]

    state_path = tmp_path / "agent-state.json"
    agent = ScriptedAgent(
        responses=responses,
        model=None,
        tokenizer=None,
        model_config=ModelConfig(),
        device=torch.device("cpu"),
        tools=[_tool()],
        handlers={"calculator": lambda arguments: int(arguments["value"]) * 2},
        state_path=state_path,
        max_steps=4,
    )

    state = agent.run("Double four")

    assert state.status == "completed"
    assert state.final_answer == "The result is 8."
    assert len(state.steps) == 2
    assert state.steps[0].tool_results[0]["result"] == 8
    assert "<|tool_result|>" in agent.prompts[1]
    assert '"result":8' in agent.prompts[1]
    assert state_path.exists()
    assert AgentState.load(state_path).final_answer == "The result is 8."


def test_agent_requires_handlers_for_declared_tools(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing handlers"):
        ScriptedAgent(
            responses=[],
            model=None,
            tokenizer=None,
            model_config=ModelConfig(),
            device=torch.device("cpu"),
            tools=[_tool()],
            handlers={},
            state_path=tmp_path / "state.json",
        )


def test_resume_rejects_different_objective(tmp_path: Path) -> None:
    state_path = tmp_path / "agent-state.json"
    AgentState(objective="Original task").save(state_path)
    agent = ScriptedAgent(
        responses=[],
        model=None,
        tokenizer=None,
        model_config=ModelConfig(),
        device=torch.device("cpu"),
        tools=[],
        handlers={},
        state_path=state_path,
    )

    with pytest.raises(ValueError, match="does not match"):
        agent.run("Different task", resume=True)
