"""Trace object — captures every agent call and tool invocation for demo visibility."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_active_trace: ContextVar["Trace | None"] = ContextVar("active_trace", default=None)


@dataclass
class ToolCall:
    tool: str
    query: str
    result: Any
    latency_ms: float


@dataclass
class AgentCall:
    agent: str
    step: str  # e.g. "initial", "reflection", "judge-rule-0"
    model: str
    system_prompt: str
    user_prompt: str
    response: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Trace:
    calls: list[AgentCall] = field(default_factory=list)

    def add(self, call: AgentCall) -> None:
        self.calls.append(call)

    def add_tool(self, tool_call: ToolCall) -> None:
        if self.calls:
            self.calls[-1].tool_calls.append(tool_call)

    def for_agent(self, agent: str) -> list[AgentCall]:
        return [c for c in self.calls if c.agent == agent]

    @property
    def total_tokens(self) -> tuple[int, int]:
        return (sum(c.tokens_in for c in self.calls), sum(c.tokens_out for c in self.calls))


def current_trace() -> Trace | None:
    return _active_trace.get()


class trace_context:
    """Context manager that installs a fresh Trace for the duration of a block."""

    def __init__(self) -> None:
        self.trace = Trace()
        self._token: Any = None

    def __enter__(self) -> Trace:
        self._token = _active_trace.set(self.trace)
        return self.trace

    def __exit__(self, *_exc: Any) -> None:
        _active_trace.reset(self._token)
