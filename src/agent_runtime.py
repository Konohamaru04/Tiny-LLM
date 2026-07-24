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
    """Iterative, resumable tool-use runtime for long-horizon tasks.

    The runtime keeps the objective stable, feeds tool results back using the
    same protocol used during SFT, and retains only a configurable recent trace
    in the active prompt. The complete trajectory remains persisted on disk.
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
        trace_window_steps: int = 8,
        state_path: str | Path = "runs/agent_state.json",
    ):
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if trace_window_steps <= 0:
            raise ValueError("trace_window_steps must be > 0")

        tool_names = {tool.name for tool in tools}
        if len(tool_names) != len(tools):
            raise ValueError("Tool definitions must have unique names")
        missing_handlers = sorted(tool_names - set(handlers))
        if missing_handlers:
            raise ValueError(
                "Missing handlers for tools: " + ", ".join(missing_handlers)
            )

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
        self.trace_window_steps = trace_window_steps
        self.state_path = Path(state_path)

    def _decode(self, prompt_tokens: list[int], output_ids: torch.Tensor) -> str:
        generated = output_ids[0, len(prompt_tokens) :].tolist()
        return self.tokenizer.decode(generated).strip()

    def _compact_trace(self, state: AgentState) -> str:
        blocks = [f"OBJECTIVE:\n{state.objective}"]
        recent_steps = state.steps[-self.trace_window_steps :]

        if len(state.steps) > len(recent_steps):
            blocks.append(
                f"EARLIER PROGRESS:\n{len(state.steps) - len(recent_steps)} completed "
                "steps are stored in the persisted task state."
            )

        for step in recent_steps:
            blocks.append(f"STEP {step.index} MODEL OUTPUT:\n{step.model_output}")
            formatted_results = [
                str(result.get("formatted", "")).strip()
                for result in step.tool_results
                if str(result.get("formatted", "")).strip()
            ]
            if formatted_results:
                blocks.append("TOOL RESULTS:\n" + "\n".join(formatted_results))

        if state.steps:
            blocks.append(
                "Continue from the tool results above. Emit another structured tool "
                "call when work remains, or emit <|final|> followed by the completed answer."
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
        normalized_objective = objective.strip()
        if not normalized_objective:
            raise ValueError("objective must not be empty")

        if resume and self.state_path.exists():
            state = AgentState.load(self.state_path)
            if state.objective != normalized_objective:
                raise ValueError(
                    "Resume objective does not match the persisted agent objective"
                )
            if state.status == "completed":
                return state
            state.status = "running"
        else:
            state = AgentState(objective=normalized_objective)

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
                    result = {
                        "name": call.name,
                        "id": call.call_id,
                        "ok": True,
                        "result": value,
                    }
                    protocol_value: Any = value
                except Exception as exc:
                    result = {
                        "name": call.name,
                        "id": call.call_id,
                        "ok": False,
                        "error": str(exc),
                    }
                    protocol_value = {"ok": False, "error": str(exc)}
                result["formatted"] = format_tool_result(
                    call.name,
                    protocol_value,
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
