from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from src.capabilities import (
    ParsedAssistantResponse,
    ThinkingMode,
    ToolDefinition,
    format_tool_result,
    parse_assistant_response,
    validate_tool_calls,
)
from src.config import ModelConfig
from src.generation import build_chat_prompt_tokens, generate
from src.tokenizer_utils import SentencePieceTokenizer


ToolHandler = Callable[[Mapping[str, Any]], Any]


@dataclass
class AgentStep:
    index: int
    prompt: str
    model_output: str
    thinking: str = ""
    final: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentState:
    objective: str
    status: str = "running"
    final_answer: str = ""
    steps: list[AgentStep] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AgentState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["steps"] = [AgentStep(**step) for step in payload.get("steps", [])]
        return cls(**payload)


class LongHorizonAgent:
    """Iterative tool-use runtime for long-horizon tasks.

    The model is repeatedly called with compacted prior observations. State is
    persisted after every step so interrupted tasks can resume. This runtime
    supplies orchestration; actual long-horizon quality still depends on the
    unified checkpoint being trained on multi-step trajectories.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: SentencePieceTokenizer,
        model_config: ModelConfig,
        device: torch.device,
        tools: Sequence[ToolDefinition],
        handlers: Mapping[str, ToolHandler],
        *,
        system_prompt: str = (
            "You are a careful autonomous assistant. Work step by step, use tools "
            "when needed, inspect results, and only finish when the objective is complete."
        ),
        max_steps: int = 16,
        max_new_tokens: int = 768,
        temperature: float = 0.3,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        state_path: str | Path = "runs/agent_state.json",
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.device = device
        self.tools = list(tools)
        self.handlers = dict(handlers)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.state_path = Path(state_path)
        validate_tool_calls([], self.handlers)

    def _decode(self, prompt_tokens: list[int], output_ids: torch.Tensor) -> str:
        generated = output_ids[0, len(prompt_tokens) :].tolist()
        return self.tokenizer.decode(generated).strip()

    def _compact_trace(self, state: AgentState) -> str:
        if not state.steps:
            return state.objective
        blocks = [f"OBJECTIVE:\n{state.objective}"]
        for step in state.steps:
            blocks.append(f"STEP {step.index} MODEL OUTPUT:\n{step.model_output}")
            if step.tool_results:
                blocks.append(
                    "OBSERVATIONS:\n"
                    + "\n".join(
                        json.dumps(result, ensure_ascii=False)
                        for result in step.tool_results
                    )
                )
        blocks.append(
            "Continue from the observations above. Use another tool call or emit "
            "<|final|> followed by the completed answer."
        )
        return "\n\n".join(blocks)

    def _run_model(self, user_message: str) -> ParsedAssistantResponse:
        prompt_tokens = build_chat_prompt_tokens(
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            history=[],
            user_message=user_message,
            block_size=self.model_config.block_size,
            max_history_turns=0,
            thinking_mode=ThinkingMode.AUTO,
            tools=self.tools,
        )
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        output_ids = generate(
            model=self.model,
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=[self.tokenizer.eos_id],
        )
        return parse_assistant_response(self._decode(prompt_tokens, output_ids))

    def run(self, objective: str, *, resume: bool = False) -> AgentState:
        state = (
            AgentState.load(self.state_path)
            if resume and self.state_path.exists()
            else AgentState(objective=objective.strip())
        )
        if not state.objective:
            raise ValueError("objective must not be empty")

        for step_index in range(len(state.steps) + 1, self.max_steps + 1):
            prompt = self._compact_trace(state)
            parsed = self._run_model(prompt)
            validate_tool_calls(parsed.tool_calls, self.handlers)
            step = AgentStep(
                index=step_index,
                prompt=prompt,
                model_output=parsed.raw_text,
                thinking=parsed.thinking,
                final=parsed.final,
                tool_calls=[
                    {"name": call.name, "arguments": dict(call.arguments), "id": call.call_id}
                    for call in parsed.tool_calls
                ],
            )

            for call in parsed.tool_calls:
                try:
                    value = self.handlers[call.name](call.arguments)
                    result = {"name": call.name, "id": call.call_id, "ok": True, "result": value}
                except Exception as exc:  # Tool failures become observations for recovery.
                    result = {"name": call.name, "id": call.call_id, "ok": False, "error": str(exc)}
                result["formatted"] = format_tool_result(
                    call.name,
                    result.get("result", result.get("error")),
                    call.call_id,
                )
                step.tool_results.append(result)

            state.steps.append(step)
            if parsed.final and not parsed.tool_calls:
                state.status = "completed"
                state.final_answer = parsed.final
                state.save(self.state_path)
                return state
            state.save(self.state_path)

        state.status = "max_steps_reached"
        state.save(self.state_path)
        return state
